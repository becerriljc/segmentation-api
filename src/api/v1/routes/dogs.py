from fastapi import APIRouter, HTTPException, File, UploadFile

from ..models.dogs import get_image_filtered

router = APIRouter(
    prefix='/dogs',
    tags=["dogs"],
    responses={
        404: {
            'description': 'No encontrado'
        }
    }
)

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="El archivo proporcionado no es una imagen.")

    try:
        result = await get_image_filtered(file)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
