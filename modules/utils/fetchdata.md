## Fetch data from Civitai
## Fetch API URL
# https://civitai.com/api/v1/images?limit=200&sort=Most%20Reactions&period=Month

## Sample API Response
# {
#   "items": [
#     {
#       "id": 60535375,
#       "url": "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/2eb7f731-4136-4295-a0f4-9e140c3a6c96/width=832/2eb7f731-4136-4295-a0f4-9e140c3a6c96.jpeg",
#       "hash": "UMGj$%=x~UJR_Lt6-ow^%fWVsSoJbbR*ni$%",
#       "width": 832,
#       "height": 1216,
#       "nsfwLevel": "None",
#       "type": "image",
#       "nsfw": false,
#       "browsingLevel": 1,
#       "createdAt": "2025-02-28T15:30:00.000Z",
#       "postId": 13542615,
#       "stats": {
#         "cryCount": 495,
#         "laughCount": 969,
#         "likeCount": 6865,
#         "dislikeCount": 0,
#         "heartCount": 2499,
#         "commentCount": 8
#       },
#       "meta": {
#         "Size": "832x1216",
#         "nsfw": false,
#         "seed": 680383978,
#         "draft": false,
#         "steps": 20,
#         "width": 832,
#         "height": 1216,
#         "prompt": "A humanoid (black cat) wearing red samurai armor. He is wearing a white headband with the Japanese rising sun on it. Holding a katana He is in a field. \nFull body\nCinematic lighting, volumetric lighting, 8k quality",
#         "sampler": "Undefined",
#         "cfgScale": 7,
#         "clipSkip": 2,
#         "fluxMode": "urn:air:flux1:checkpoint:civitai:618692@691639",
#         "quantity": 1,
#         "workflow": "txt2img",
#         "baseModel": "Flux1",
#         "resources": [],
#         "Created Date": "2025-02-27T0240:27.6424173Z",
#         "civitaiResources": [
#           {
#             "type": "checkpoint",
#             "modelVersionId": 691639,
#             "modelVersionName": "Dev"
#           },
#           {
#             "type": "lora",
#             "weight": 0.5,
#             "modelVersionId": 863655,
#             "modelVersionName": "Balance_v2.0"
#           },
#           {
#             "type": "lora",
#             "weight": 0.8,
#             "modelVersionId": 747534,
#             "modelVersionName": "Flux.1 D v1"
#           },
#           {
#             "type": "lora",
#             "weight": 0.55,
#             "modelVersionId": 806265,
#             "modelVersionName": "v1.0"
#           }
#         ]
#       },
#       "username": "Entersandman",
#       "baseModel": "Flux.1 D"
#     }
#   ],
#   "metadata": {
#     "nextCursor": "1|1740756600000",
#     "nextPage": "https://civitai.com/api/v1/images?limit=1&sort=Most%20Reactions&period=Month&cursor=1%7C1740756600000"
#   }
# }

## Requirements 
## 1. Fetch data from Civitai API, need to extract id, image url, prompt and negative prompt, other meta data optional
## 2. Process data, need to do null checking and handle missing data, focus on baseModel = ["Illustrious", "Flux.1 D", "Pony"], remove no prompt record.
## 3. Store in Qdrant
## 4. Return data from Qdrant
## Need to handle duplicate records in Qdrant, need to check if record already exists before inserting
## Need to handle api limits (200 max per request), need to handle pagination and cursor, store the latest cursor value in the temp text file for next use
## API settings: limit=200, sort=Most%20Reactions, period=Month
## API security in secrets.toml
