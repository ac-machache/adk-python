# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from typing import AsyncGenerator
from typing import Union

from google.genai import live
from google.genai import types

from .base_llm_connection import BaseLlmConnection
from .llm_response import LlmResponse

logger = logging.getLogger('google_adk.' + __name__)

RealtimeInput = Union[types.Blob, types.ActivityStart, types.ActivityEnd]


class GeminiLlmConnection(BaseLlmConnection):
  """The Gemini model connection."""

  def __init__(self, gemini_session: live.AsyncSession):
    self._gemini_session = gemini_session

  async def send_history(self, history: list[types.Content]):
    """Sends the conversation history to the gemini model.

    You call this method right after setting up the model connection.
    The model will respond if the last content is from user, otherwise it will
    wait for new user input before responding.

    Args:
      history: The conversation history to send to the model.
    """

    # TODO: Remove this filter and translate unary contents to streaming
    # contents properly.

    # We ignore any audio from user during the agent transfer phase
    contents = [
        content
        for content in history
        if content.parts and content.parts[0].text
    ]

    if contents:
      await self._gemini_session.send_client_content(
          turns=contents,
          turn_complete=contents[-1].role == 'user',
      )
    else:
      logger.info('no content is sent')

  async def send_content(self, content: types.Content):
    """Sends a user content to the gemini model.

    The model will respond immediately upon receiving the content.
    If you send function responses, all parts in the content should be function
    responses.

    Args:
      content: The content to send to the model.
    """

    assert content.parts

    def _is_tool_response_content(c: types.Content) -> bool:
      return bool(c.parts) and all(p.function_response for p in c.parts)

    if _is_tool_response_content(content):
      function_responses = [part.function_response for part in content.parts]
      logger.debug('Sending LLM function response: %s', function_responses)
      await self._gemini_session.send_tool_response(
          function_responses=function_responses,
      )
    elif any(p.function_response for p in content.parts):
      # Mixed tool/non-tool parts are not supported
      raise ValueError(
          'Content parts must be either all function_response or none.'
      )
    else:
      logger.debug('Sending LLM new content %s', content)
      await self._gemini_session.send_client_content(
          turns=[content],
          turn_complete=True,
      )

  async def send_realtime(self, input: RealtimeInput):
    """Sends a chunk of audio or a frame of video to the model in realtime.

    Args:
      input: The input to send to the model.
    """
    if isinstance(input, types.Blob):
      # Route blob types to appropriate send_realtime_input parameters
      if input.mime_type.startswith('audio/'):
        logger.debug('Sending LLM realtime audio blob')
        await self._gemini_session.send_realtime_input(audio=input)
      elif input.mime_type.startswith('video/'):
        logger.debug('Sending LLM realtime video blob')
        await self._gemini_session.send_realtime_input(video=input)
      elif input.mime_type.startswith('image/'):
        logger.debug('Sending LLM realtime image blob')
        await self._gemini_session.send_realtime_input(media=input)
      else:
        # Fallback for unknown blob types
        logger.debug(
            'Sending LLM realtime blob with media parameter: %s',
            input.mime_type,
        )
        await self._gemini_session.send_realtime_input(media=input)
    elif isinstance(input, types.ActivityStart):
      logger.debug('Sending LLM activity start signal')
      await self._gemini_session.send_realtime_input(activity_start=input)
    elif isinstance(input, types.ActivityEnd):
      logger.debug('Sending LLM activity end signal')
      await self._gemini_session.send_realtime_input(activity_end=input)
    else:
      raise ValueError('Unsupported input type: %s' % type(input))

  def __build_full_text_response(self, text: str):
    """Builds a full text response.

    The text should not partial and the returned LlmResponse is not be
    partial.

    Args:
      text: The text to be included in the response.

    Returns:
      An LlmResponse containing the full text.
    """
    return LlmResponse(
        content=types.Content(
            role='model',
            parts=[types.Part.from_text(text=text)],
        ),
    )

  async def _handle_model_turn(
      self, content, interrupted: bool, text_buffer: list[str]
  ) -> AsyncGenerator[LlmResponse, None]:
    if content and content.parts:
      llm_response = LlmResponse(content=content, interrupted=interrupted)
      if content.parts[0].text:
        text_buffer[0] += content.parts[0].text
        llm_response.partial = True
      elif content.parts[0].inline_data:
        # Inline media (e.g., audio frames) should be surfaced but not
        # persisted to session history. Mark as partial to prevent saves.
        llm_response.partial = True
      # don't yield the merged text event when receiving audio data
      elif text_buffer[0] and not content.parts[0].inline_data:
        yield self.__build_full_text_response(text_buffer[0])
        text_buffer[0] = ''
      yield llm_response

  async def _handle_input_transcription(
      self, input_transcription, user_text_buffer: list[str]
  ) -> AsyncGenerator[LlmResponse, None]:
    if input_transcription and input_transcription.text:
      user_text = input_transcription.text
      user_text_buffer[0] += user_text
      parts = [
          types.Part.from_text(
              text=user_text,
          )
      ]
      yield LlmResponse(
          content=types.Content(role='user', parts=parts),
          partial=True,
      )

  async def _handle_output_transcription(
      self, output_transcription, text_buffer: list[str]
  ) -> AsyncGenerator[LlmResponse, None]:
    if output_transcription and output_transcription.text:
      # TODO: Right now, we just support output_transcription without
      # changing interface and data protocol. Later, we can consider to
      # support output_transcription as a separate field in LlmResponse.

      # Transcription is always considered as partial event
      # We rely on other control signals to determine when to yield the
      # full text response(turn_complete, interrupted, or tool_call).
      text_buffer[0] += output_transcription.text
      parts = [types.Part.from_text(text=output_transcription.text)]
      yield LlmResponse(
          content=types.Content(role='model', parts=parts), partial=True
      )

  async def _handle_tool_call(
      self, tool_call, text_buffer: list[str]
  ) -> AsyncGenerator[LlmResponse, None]:
    if text_buffer[0]:
      yield self.__build_full_text_response(text_buffer[0])
      text_buffer[0] = ''
    parts = [
        types.Part(function_call=function_call)
        for function_call in tool_call.function_calls
    ]
    yield LlmResponse(content=types.Content(role='model', parts=parts))

  async def _process_model_content(
      self, message, text_buffer: list[str]
  ) -> AsyncGenerator[LlmResponse, None]:
    """Handles model-generated content like text, media, and tool calls."""
    if message.server_content and message.server_content.model_turn:
      async for response in self._handle_model_turn(
          message.server_content.model_turn,
          message.server_content.interrupted,
          text_buffer,
      ):
        yield response

    if message.tool_call:
      async for response in self._handle_tool_call(
          message.tool_call, text_buffer
      ):
        yield response

  async def _process_transcriptions(
      self,
      server_content,
      text_buffer: list[str],
      user_text_buffer: list[str],
  ) -> AsyncGenerator[LlmResponse, None]:
    """Handles input and output transcriptions."""
    if not server_content:
      return

    async for response in self._handle_input_transcription(
        server_content.input_transcription, user_text_buffer
    ):
      yield response

    async for response in self._handle_output_transcription(
        server_content.output_transcription, text_buffer
    ):
      yield response

  async def _process_session_and_control_events(
      self, message, text_buffer: list[str]
  ) -> AsyncGenerator[LlmResponse, None]:
    """Handles session management and control signals from the server."""
    # Session Management Events
    session_resumption_update = getattr(
        message, 'session_resumption_update', None
    )
    if session_resumption_update:
      logger.info(
          'Received session resumption update: %s', session_resumption_update
      )
      yield LlmResponse(
          live_session_resumption_update=session_resumption_update
      )

    go_away = getattr(message, 'go_away', None) or getattr(
        message, 'goaway', None
    )
    if go_away:
      yield LlmResponse(
          goaway=go_away,
          partial=True,
      )

    # Control Events (excluding turn_complete)
    if message.server_content:
      generation_complete = getattr(
          message.server_content, 'generation_complete', False
      )
      if generation_complete:
        if text_buffer[0]:
          yield self.__build_full_text_response(text_buffer[0])
          text_buffer[0] = ''
        yield LlmResponse(
            generation_complete=True,
            partial=True,
        )

      if message.server_content.interrupted and text_buffer[0]:
        yield self.__build_full_text_response(text_buffer[0])
        text_buffer[0] = ''
      # This yields an LlmResponse with interrupted=True for the event,
      # or with interrupted=False/None which is filtered out downstream.
      yield LlmResponse(interrupted=message.server_content.interrupted)

  async def receive(self) -> AsyncGenerator[LlmResponse, None]:
    """Receives the model response using the llm server connection.

    Yields:
      LlmResponse: The model response.
    """

    text_buffer = ['']
    user_text_buffer = ['']
    async for message in self._gemini_session.receive():
      logger.debug('Got LLM Live message: %s', message)

      # First, check if the model is replying to commit the user's utterance.
      content = (
          message.server_content.model_turn if message.server_content else None
      )
      model_turn_has_content = content and content.parts
      model_is_replying = (
          message.tool_call
          or (
              message.server_content
              and message.server_content.output_transcription
          )
          or model_turn_has_content
      )
      if user_text_buffer[0] and model_is_replying:
        yield LlmResponse(
            content=types.Content(
                role='user',
                parts=[types.Part.from_text(text=user_text_buffer[0])],
            )
        )
        user_text_buffer[0] = ''

      # Process events in logical groups.
      async for response in self._process_model_content(message, text_buffer):
        yield response

      async for response in self._process_transcriptions(
          message.server_content, text_buffer, user_text_buffer
      ):
        yield response

      async for response in self._process_session_and_control_events(
          message, text_buffer
      ):
        yield response

      # Handle the terminal event last.
      if message.server_content and message.server_content.turn_complete:
        if text_buffer[0]:
          yield self.__build_full_text_response(text_buffer[0])
          text_buffer[0] = ''
        yield LlmResponse(
            turn_complete=True,
            interrupted=message.server_content.interrupted,
            usage_metadata=getattr(
                message.server_content, 'usage_metadata', None
            ),
        )
        # A turn_complete event signals the end of the interaction for now.
        return

  async def close(self):
    """Closes the llm server connection."""

    await self._gemini_session.close()
