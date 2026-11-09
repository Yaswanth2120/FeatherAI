"""Buffer bounded request bodies before routing, including chunked transfers."""
from fastapi.responses import JSONResponse


class BodyLimitMiddleware:
    def __init__(self, app, limit=65536):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        messages, total = [], 0
        while True:
            message = await receive()
            if message['type'] == 'http.disconnect':
                return
            total += len(message.get('body', b''))
            if total > self.limit:
                response = JSONResponse({'detail': 'Request too large'}, status_code=413)
                return await response(scope, receive, send)
            messages.append(message)
            if not message.get('more_body', False):
                break
        position = 0
        async def bounded_receive():
            nonlocal position
            if position < len(messages):
                item = messages[position]
                position += 1
                return item
            return await receive()
        await self.app(scope, bounded_receive, send)
