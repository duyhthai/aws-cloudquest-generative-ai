from dispatchers import utils
from sm_utils.sm_langchain_sample import SagemakerLangchainBot
import json
import os
import logging

logger = utils.get_logger(__name__)
logger.setLevel(logging.DEBUG)
CHAT_HISTORY="chat_history"
initial_history = {CHAT_HISTORY: f"AI: Hi there! How Can I help you?\nHuman: ",}
endpoint_name = os.environ['ENDPOINT_NAME']

class LexV2SMLangchainDispatcher():

    def __init__(self, intent_request):
        # See lex bot input format to lambda https://docs.aws.amazon.com/lex/latest/dg/lambda-input-response-format.html
        self.intent_request = intent_request
        self.localeId = self.intent_request['bot']['localeId']
        self.input_transcript = self.intent_request['inputTranscript'] # user input
        self.session_attributes = utils.get_session_attributes(
            self.intent_request)
        self.fulfillment_state = "Fulfilled"
        self.text = "" # response from endpoint
        self.message = {'contentType': 'PlainText','content': self.text}
        print("In LexV2SMLangchainDispatcher __init__")
        print(intent_request)

    def dispatch_intent(self):

        
        # # define prompt
        prompt_template = """Give a fact based description of the corporation in the input below. The answer must come from reliable sources such as Wikipedia. 
        
        Chat History:
        {chat_history}

        Conversation:
        Human: {input}
        AI:"""
        

        # Set context with convo history for custom memory "RAG" in langchain
        conv_context: str = self.session_attributes.get('ConversationContext',json.dumps(initial_history))

        # LLM
        langchain_bot = SagemakerLangchainBot(
            prompt_template = prompt_template,
            sm_endpoint_name = endpoint_name,
            region_name = os.environ.get('AWS_REGION',"us-east-1"),
            lex_conv_history = conv_context
        )
    
        llm_response = langchain_bot.call_llm(user_input=self.input_transcript)
        

        self.message = {
            'contentType': 'PlainText',
            'content': llm_response
        }

        # save chat history as Lex session attributes
        session_conv_context = json.loads(conv_context)
        session_conv_context[CHAT_HISTORY] = session_conv_context[CHAT_HISTORY] + self.input_transcript + f"\nAI: {llm_response}" +"\nHuman: "
        self.session_attributes["ConversationContext"] = json.dumps(session_conv_context)

        self.response = utils.close(
            self.intent_request, 
            self.session_attributes, 
            self.fulfillment_state, 
            self.message
        )

        return self.response