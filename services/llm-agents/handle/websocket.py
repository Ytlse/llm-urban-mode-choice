from typing import List, Optional, Callable

from loguru import logger
import asyncio
import websockets
import json

class WebSocketClient:
    def __init__(self, uri: str, reconnect_delay: int = 5):
        self.uri = uri
        self.reconnect_delay = reconnect_delay
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.running = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = float('inf')  # indefinite reconnection until GAMA is available
        
        # Callbacks
        self.on_message: Optional[Callable] = None
        self.on_connect: Optional[Callable] = None
        self.on_disconnect: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

    async def connect(self):
        """Connect to the WebSocket server"""
        try:
            logger.info(f"Connecting to {self.uri}...")
            
            # ping_timeout must tolerate occasional event-loop stalls (CPU
            # computations, STM reflections): at 10s, a ~15s stall was enough to close the
            # socket (1006) and a burst of pushes went into the void (run 2026-07-08:
            # 339 agents recovered by the watchdog). 60s only delays the detection of a
            # real cut by ~1 min, already covered by the arrival watchdog.
            self.websocket = await websockets.connect(
                self.uri,
                ping_interval=20,
                ping_timeout=60,
                close_timeout=20,
                max_size=10**7,  # 10MB max message size
                compression=None  # Disable compression to improve performance
            )
            
            self.reconnect_attempts = 0
            logger.info(f"✅ WebSocket connected to {self.uri}")
            
            if self.on_connect:
                await self.on_connect()
                
            return True
            
        except Exception as e:
            logger.info(f"Connection failed: {e}")
            if self.on_error:
                await self.on_error(e)
            return False

    async def disconnect(self):
        """Ngắt kết nối"""
        self.running = False
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            logger.info("Disconnected from WebSocket")
            
            if self.on_disconnect:
                await self.on_disconnect()

    async def send_message(self, message: str):
        """Send a message"""
        if self.websocket:
            try:
                await self.websocket.send(message)
                return True
            except Exception as e:
                logger.error(f"Send failed: {e}")
                return False
        else:
            logger.warning("WebSocket not connected, cannot send message")
            return False

    async def send_json(self, data: dict):
        """Send JSON data"""
        return await self.send_message(json.dumps(data))

    async def listen(self):
        """Listen for messages from the server"""
        try:
            while self.running and self.websocket:
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(), 
                        timeout=30.0
                    )
                    
                    #logger.debug(f"Received: {message}")
                    
                    if self.on_message:
                        await self.on_message(message)
                        
                except asyncio.TimeoutError:
                    # Normal timeout, continue the loop
                    continue
                    
                except websockets.exceptions.ConnectionClosed as e:
                    logger.warning(f"Connection closed: {e.code} - {e.reason}")
                    break
                    
                except Exception as e:
                    logger.exception(f"Listen error: {e}")
                    break
                    
        except Exception as e:
            logger.error(f"Listen loop error: {e}")

    async def run_with_reconnect(self):
        """Run the client with auto reconnect"""
        self.running = True

        while self.running:
            try:
                if await self.connect():
                    await self.listen()

                if self.running and self.reconnect_attempts < self.max_reconnect_attempts:
                    self.reconnect_attempts += 1
                    logger.info(f"Reconnecting in {self.reconnect_delay}s... (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})")
                    await asyncio.sleep(self.reconnect_delay)
                elif self.reconnect_attempts >= self.max_reconnect_attempts:
                    logger.error("Max reconnect attempts reached. Stopping.")
                    break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Run loop error: {e}")
                if self.running:
                    await asyncio.sleep(self.reconnect_delay)

    async def stop(self):
        """Stop the client"""
        logger.info("Stopping WebSocket client...")
        await self.disconnect()
