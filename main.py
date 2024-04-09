# ------------------------------------------------------------------------------------------------------------#
#                                                 Libraries                                                   #
# ------------------------------------------------------------------------------------------------------------#

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

# ------------------------------------------------------------------------------------------------------------#
#                                                Dependencies                                                 #
# ------------------------------------------------------------------------------------------------------------#

from dependencies import get_auth_header

# ------------------------------------------------------------------------------------------------------------#
#                                                    Routes                                                   #
# ------------------------------------------------------------------------------------------------------------#
from routes import dogs

# ------------------------------------------------------------------------------------------------------------#
#                                          Constants and Variables                                            #
# ------------------------------------------------------------------------------------------------------------#

baseUrl = '/api/v1/segmentation'

app = FastAPI(
    dependencies=[Depends(get_auth_header)],
)

# ------------------------------------------------------------------------------------------------------------#
#                                                Development                                                  #
# ------------------------------------------------------------------------------------------------------------#

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    dogs.router,
    prefix=baseUrl,
    tags=['segmentation'],
)