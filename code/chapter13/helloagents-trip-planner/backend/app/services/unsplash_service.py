"""景点图片服务：优先维基百科配图，Unsplash 英文关键词兜底"""

import re
import requests
from typing import List, Optional, Dict
from urllib.parse import quote
from ..config import get_settings


# 常见景点中文名 -> Unsplash 更准的英文检索词
LANDMARK_EN_QUERIES: Dict[str, str] = {
    "故宫": "Forbidden City Beijing palace",
    "故宫博物院": "Forbidden City Beijing palace",
    "紫禁城": "Forbidden City Beijing",
    "天安门": "Tiananmen Square Beijing",
    "天安门广场": "Tiananmen Square Beijing",
    "长城": "Great Wall of China",
    "八达岭长城": "Great Wall of China Badaling",
    "慕田峪长城": "Great Wall Mutianyu",
    "颐和园": "Summer Palace Beijing",
    "天坛": "Temple of Heaven Beijing",
    "天坛公园": "Temple of Heaven Beijing",
    "圆明园": "Yuanmingyuan Old Summer Palace",
    "北海公园": "Beihai Park Beijing",
    "景山公园": "Jingshan Park Beijing",
    "雍和宫": "Yonghe Temple Lama Temple Beijing",
    "国子监": "Guozijian Imperial College Beijing",
    "孔庙": "Confucius Temple Beijing",
    "北京孔庙": "Confucius Temple Beijing",
    "中国国家博物馆": "National Museum of China Beijing",
    "国家博物馆": "National Museum of China",
    "鸟巢": "Beijing National Stadium Bird Nest",
    "国家体育场": "Beijing National Stadium",
    "水立方": "Beijing National Aquatics Center",
    "什刹海": "Shichahai Beijing hutong",
    "南锣鼓巷": "Nanluoguxiang Beijing",
    "恭王府": "Prince Gong Mansion Beijing",
    "明十三陵": "Ming Tombs Beijing",
    "香山": "Fragrant Hills Beijing",
    "香山公园": "Fragrant Hills Beijing",
    "清华大学": "Tsinghua University campus",
    "北京大学": "Peking University campus",
    "鼓楼": "Drum Tower Beijing",
    "钟鼓楼": "Drum Bell Tower Beijing",
    "钟楼": "Bell Tower Beijing",
    "外滩": "The Bund Shanghai",
    "东方明珠": "Oriental Pearl Tower Shanghai",
    "西湖": "West Lake Hangzhou",
    "兵马俑": "Terracotta Army Xi'an",
    "秦始皇兵马俑": "Terracotta Army Xi'an",
    "大雁塔": "Giant Wild Goose Pagoda Xi'an",
    "黄山": "Huangshan Yellow Mountain",
    "张家界": "Zhangjiajie National Forest Park",
    "九寨沟": "Jiuzhaigou Valley",
    "布达拉宫": "Potala Palace Lhasa",
    "宽窄巷子": "Kuanzhai Alley Chengdu",
    "大熊猫基地": "Chengdu Research Base of Giant Panda",
    "成都大熊猫繁育研究基地": "Chengdu Research Base of Giant Panda",
}

# 中文名 -> 英文维基标题（配图通常更准）
LANDMARK_EN_WIKI: Dict[str, str] = {
    "故宫": "Forbidden City",
    "故宫博物院": "Forbidden City",
    "紫禁城": "Forbidden City",
    "天安门": "Tiananmen",
    "天安门广场": "Tiananmen Square",
    "颐和园": "Summer Palace",
    "天坛": "Temple of Heaven",
    "天坛公园": "Temple of Heaven",
    "雍和宫": "Yonghe Temple",
    "国子监": "Guozijian",
    "孔庙": "Beijing Temple of Confucius",
    "北京孔庙": "Beijing Temple of Confucius",
    "中国国家博物馆": "National Museum of China",
    "国家博物馆": "National Museum of China",
    "鸟巢": "Beijing National Stadium",
    "国家体育场": "Beijing National Stadium",
    "水立方": "Beijing National Aquatics Center",
    "恭王府": "Prince Gong Mansion",
    "外滩": "The Bund",
    "东方明珠": "Oriental Pearl Tower",
    "西湖": "West Lake",
    "兵马俑": "Terracotta Army",
    "布达拉宫": "Potala Palace",
}


class UnsplashService:
    """景点图片服务（兼容原类名）"""

    def __init__(self):
        settings = get_settings()
        self.access_key = settings.unsplash_access_key
        self.base_url = "https://api.unsplash.com"
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "HelloAgentsTripPlanner/1.0 (educational; contact=local)"
        })

    def _candidate_titles(self, name: str) -> List[str]:
        """生成可能的维基条目标题"""
        name = (name or "").strip()
        if not name:
            return []

        titles = [name]
        # 孔庙（北京孔庙） / 孔庙(北京孔庙)
        m = re.match(r"^(.+?)\s*[（(](.+?)[）)]\s*$", name)
        if m:
            titles.extend([m.group(1).strip(), m.group(2).strip()])

        # 去掉常见后缀再试
        for suffix in ("博物院", "博物馆", "公园", "风景区", "景区", "遗址"):
            if name.endswith(suffix) and len(name) > len(suffix) + 1:
                titles.append(name[: -len(suffix)])

        # 去重保序
        seen = set()
        result = []
        for t in titles:
            if t and t not in seen:
                seen.add(t)
                result.append(t)
        return result

    def _is_good_image_url(self, source: Optional[str], name: str = "") -> bool:
        if not source:
            return False
        lower = source.lower()
        if ".svg" in lower:
            return False
        if any(bad in lower for bad in ("logo", "wordmark", "icon", "coat_of_arms")):
            return False

        # 名称与图片文件名明显冲突时丢弃（如国博配成天安门广场图）
        if name and any(k in name for k in ("博物馆", "博物院", "Museum", "museum")):
            if "tiananmen" in lower and "museum" not in lower:
                return False
        if name and any(k in name for k in ("故宫", "Forbidden")):
            if "tiananmen_square" in lower and "forbidden" not in lower:
                return False
        return True

    def _lookup_mapped(self, name: str, mapping: Dict[str, str]) -> Optional[str]:
        for title in self._candidate_titles(name):
            if title in mapping:
                return mapping[title]
            for key, value in mapping.items():
                if key in title or title in key:
                    return value
        return None

    def _fetch_wiki_summary_image(self, lang: str, title: str, name: str = "") -> Optional[str]:
        try:
            url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title, safe='')}"
            resp = self._session.get(url, timeout=8)
            if resp.status_code != 200:
                return None
            data = resp.json()
            if data.get("type") == "disambiguation":
                return None
            for key in ("originalimage", "thumbnail"):
                source = (data.get(key) or {}).get("source")
                if self._is_good_image_url(source, name or title):
                    return source
        except Exception as e:
            print(f"⚠️ 维基配图失败 [{lang}:{title}]: {e}")
        return None

    def get_wikipedia_photo(self, name: str) -> Optional[str]:
        """从中文/英文维基百科摘要页取配图"""
        # 1) 有英文映射的景点优先英文维基（头图通常更准）
        en_title = self._lookup_mapped(name, LANDMARK_EN_WIKI)
        if en_title:
            source = self._fetch_wiki_summary_image("en", en_title, name)
            if source:
                return source

        # 2) 中文标题
        for title in self._candidate_titles(name):
            source = self._fetch_wiki_summary_image("zh", title, name)
            if source:
                return source

        # 3) opensearch 兜底
        try:
            search_url = "https://zh.wikipedia.org/w/api.php"
            resp = self._session.get(
                search_url,
                params={
                    "action": "opensearch",
                    "search": name,
                    "limit": 1,
                    "namespace": 0,
                    "format": "json",
                },
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()
            if len(data) >= 2 and data[1]:
                found = data[1][0]
                if found and found not in self._candidate_titles(name):
                    source = self._fetch_wiki_summary_image("zh", found, name)
                    if source:
                        return source
        except Exception as e:
            print(f"⚠️ 维基搜索失败: {e}")

        return None

    def _english_query(self, name: str, city: str = "") -> Optional[str]:
        """把中文景点名映射成英文检索词"""
        mapped = self._lookup_mapped(name, LANDMARK_EN_QUERIES)
        if mapped:
            return mapped

        city = (city or "").strip()
        if city:
            return f"{name} {city} landmark travel"
        return f"{name} China landmark travel"

    def search_photos(self, query: str, per_page: int = 5) -> List[dict]:
        """搜索 Unsplash 图片"""
        if not self.access_key:
            return []

        try:
            url = f"{self.base_url}/search/photos"
            params = {
                "query": query,
                "per_page": per_page,
                "orientation": "landscape",
                "client_id": self.access_key,
            }

            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            results = data.get("results", [])

            photos = []
            for photo in results:
                photos.append({
                    "id": photo.get("id"),
                    "url": photo.get("urls", {}).get("regular"),
                    "thumb": photo.get("urls", {}).get("thumb"),
                    "description": photo.get("description") or photo.get("alt_description") or "",
                    "photographer": photo.get("user", {}).get("name"),
                })

            return photos

        except Exception as e:
            print(f"❌ Unsplash搜索失败: {str(e)}")
            return []

    def _score_photo(self, photo: dict, keywords: List[str]) -> int:
        text = (photo.get("description") or "").lower()
        score = 0
        for kw in keywords:
            kw = (kw or "").strip().lower()
            if kw and kw in text:
                score += 3
        # 惩罚明显无关的通用词占主导且无关键词命中
        return score

    def get_photo_url(self, query: str) -> Optional[str]:
        """兼容旧接口：按查询词取一张图"""
        photos = self.search_photos(query, per_page=5)
        if photos:
            return photos[0].get("url")
        return None

    def get_attraction_photo(self, name: str, city: str = "") -> Optional[str]:
        """
        按景点名获取更匹配的图片：
        1) 维基百科配图（中文景点最准）
        2) Unsplash 英文映射检索 + 描述相关度排序
        """
        name = (name or "").strip()
        if not name:
            return None

        wiki = self.get_wikipedia_photo(name)
        if wiki:
            print(f"🖼️ 维基配图命中: {name}")
            return wiki

        en_query = self._english_query(name, city)
        photos = self.search_photos(en_query, per_page=8) if en_query else []

        keywords = []
        for part in re.split(r"[\s,]+", en_query or ""):
            if len(part) > 2 and part.lower() not in {"the", "and", "of", "china", "travel", "landmark"}:
                keywords.append(part)

        if photos:
            ranked = sorted(
                photos,
                key=lambda p: self._score_photo(p, keywords),
                reverse=True,
            )
            best = ranked[0]
            print(f"🖼️ Unsplash配图: {name} <- {en_query}")
            return best.get("url")

        # 最后再试一次原始中文名
        return self.get_photo_url(name)


_unsplash_service = None


def get_unsplash_service() -> UnsplashService:
    """获取图片服务实例(单例模式)"""
    global _unsplash_service

    if _unsplash_service is None:
        _unsplash_service = UnsplashService()

    return _unsplash_service
