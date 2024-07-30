import streamlit as st
import boto3
import json

# Initialize AWS clients
aws_region = 'us-east-1'

client = boto3.client("bedrock-runtime", region_name=aws_region,
)
kb_client = boto3.client('bedrock-agent-runtime', region_name=aws_region,
)


def get_kb_id():
    bedrock_agent_client = boto3.client("bedrock-agent","us-east-1",
    )

    response = bedrock_agent_client.list_knowledge_bases(maxResults=1)
    if len(response["knowledgeBaseSummaries"]) == 1:
        return response["knowledgeBaseSummaries"][0]["knowledgeBaseId"]
    else:
        return "None"


# Streamlit page configuration
st.set_page_config(page_title="A Sommelier LLM chatbot with KB", page_icon=":robot:")

# Display the wine image
st.image("winebot.png", use_column_width=True)

# Application title
st.write("""
# Sammy the Sommelier

An LLM based wine expert!
""")

kb_on = st.radio("**Enable Knowledge Base:**", ["No", "Yes"])
if kb_on == "Yes":
    kb_id = get_kb_id()
    country_in = st.selectbox("**Choose a Country:**", ["India", "Georgia", "Romania", "Uruguay", "Brazil"])
    points_in = st.slider("**Choose a minimum wine point score:**", min_value=0, max_value=100, step=5, value=50)
    price_in = st.slider("**Choose a maximum price:**", min_value=0, max_value=1000, step=10, value=1000)
# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['text'])


def invoke_llm_with_history(user_query, chat_history, kb_flag):
    body = {}

    parameters = {
        "maxTokenCount": 1000,
        "stopSequences": [],
        "temperature": 0,
        "topP": 1
    }
    conversation = "\n".join([f"{msg['role']}: {msg['text']}" for msg in chat_history])

    if kb_flag == "Yes":
        print("KbYes")
        kb_response, kb_source = invoke_kb(user_query, country_in, points_in, price_in)
        if kb_response:
            st.session_state.chat_history.append({"role": 'user', "text": user_query})
            st.session_state.chat_history.append({"role": 'kb', "text": kb_response})

        prompt = f"""
        You are a wine expert. You enjoy discussing wine in general as well as giving detailed wine recommendations .

        You are given the following query from the user:
        {user_query}
        
        You also have access to previous conversation history:
        {conversation}
        
        Take into account the following information retrieved from a knowledge base when making your recommendation:
        {kb_response}
        {price_in}
        {points_in}
        {country_in}
        
        End your response with, The source for my recommendation comes from:
        {kb_source}
        """

        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": parameters
        }).encode()

        try:
            response = client.invoke_model(
                body=body,
                modelId="amazon.titan-text-premier-v1:0",
                accept="application/json",
                contentType="application/json",
            )
            return json.loads(response["body"].read())["results"][0]["outputText"]
        except Exception as e:
            st.error(f"Error invoking LLM: {e}")
            return ""
    else:
        print("KbNo")
        prompt = f"""
        You are a wine expert. You enjoy discussing wine in general as well as giving detailed wine recommendations.


        Your previous conversation history with the human is:
        {conversation}


        Based on the question or statement below you need to recommend a wine including a brief description and price:
        {user_query}

        """

        body = json.dumps({
            "inputText": prompt,
            "textGenerationConfig": parameters
        }).encode()

        try:
            response = client.invoke_model(
                body=body,
                modelId="amazon.titan-text-premier-v1:0",
                accept="application/json",
                contentType="application/json",
            )
            return json.loads(response["body"].read())["results"][0]["outputText"]
        except Exception as e:
            st.error(f"Error invoking LLM: {e}")
            return ""


def invoke_kb(query, country, points, price):
    try:
        response = kb_client.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={
                'text': query
            },
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": 2,
                    "filter": {
                        "andAll": [
                            {
                                "equals": {"key": "country", "value": country}
                            },
                            {
                                "greaterThan": {"key": "points", "value": points}
                            },
                            {
                                "lessThan": {"key": "price", "value": price}
                            }

                        ]
                    }
                }
            }
        )

        contents = response["retrievalResults"]
        print(contents)

        wine_recommendations = ""
        source_docs = ""
        for content in contents:
            print(content['content']['text'])
            print("-------------------------------------------------------------------------------")
            print(content['location']['s3Location']['uri'])
            wine_recommendations = wine_recommendations + content['content']['text'] + "\n"
            source_docs = source_docs + content['location']['s3Location']['uri'] + "\n"
        return wine_recommendations, source_docs

    except Exception as e:
        st.error(f"Error retrieving from knowledge base: {e}")
        return ""


def main():
    user_query = st.chat_input("What kind of wine are you looking for?")
    if user_query:
        with st.chat_message('user'):
            st.markdown(user_query)

        response = invoke_llm_with_history(user_query, st.session_state.chat_history, kb_on)
        if response:
            with st.chat_message('assistant'):
                st.markdown(response)
            st.session_state.chat_history.append({"role": 'assistant', "text": response})


if __name__ == "__main__":
    main()
