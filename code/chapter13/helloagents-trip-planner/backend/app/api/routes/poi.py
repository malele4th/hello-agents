"""POI相关API路由"""

from fastapi import APIRouter, HTTPException
from ...services.unsplash_service import get_unsplash_service

router = APIRouter(prefix="/poi", tags=["POI"])


@router.get(
    "/photo",
    summary="获取景点图片",
    description="根据景点名称获取配图（优先维基百科，Unsplash 兜底）"
)
async def get_attraction_photo(name: str, city: str = ""):
    """
    获取景点图片

    Args:
        name: 景点名称
        city: 城市（可选，提升 Unsplash 检索准确度）

    Returns:
        图片URL
    """
    try:
        unsplash_service = get_unsplash_service()
        photo_url = unsplash_service.get_attraction_photo(name, city=city)

        return {
            "success": True,
            "message": "获取图片成功" if photo_url else "未找到匹配图片",
            "data": {
                "name": name,
                "photo_url": photo_url
            }
        }

    except Exception as e:
        print(f"❌ 获取景点图片失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取景点图片失败: {str(e)}"
        )
