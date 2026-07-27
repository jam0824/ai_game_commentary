import asyncio
import pyvts

plugin_info = {
    "plugin_name": "SkynaController",
    "developer": "you",
    "authentication_token_path": "./token.txt"
}

async def main():
    vts = pyvts.vts(plugin_info=plugin_info)
    await vts.connect()
    await vts.request_authenticate_token()
    await vts.request_authenticate()

    response = await vts.request(vts.vts_request.requestTrackingParameterList())
    print(response)

    await vts.close()

asyncio.run(main())