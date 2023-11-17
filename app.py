import openai
openai.api_key="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"

import pinecone
from flask import Flask, request, jsonify

# Configure Pinecone API key and index
pinecone_api_key = "48bd4df6-569e-4334-94ac-8f155d31a2f1"
pinecone_index_name = "mentor"

# Create a Pinecone client
pinecone.init(api_key=pinecone_api_key, environment="us-west4-gcp-free")
index = pinecone.Index("mentor")

app = Flask(__name__)

def search_in_pinecone(query):
    # Search for embeddings in Pinecone
    results = index.query(vector=query,top_k=1,include_metadata=True)
    return results

def chat_with_gpt(contexts,input):
    # Use OpenAI's ChatGPT to generate a response
    # build our prompt with the retrieved contexts included
    prompt_start = (
            "I'm giving you a profile for a mentor, please tell me why this is the best mentor for me, considering that I'm want to "
            + input +
            "Mentor Profile:\n"
    )
    # append contexts until hitting limit
    for i in range(0, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= 500:
            prompt = (
                    prompt_start +
                    "\n\n---\n\n".join(contexts[:i - 1])
            )
            break
        elif i == len(contexts) - 1:
            prompt = (
                    prompt_start +
                    "\n\n---\n\n".join(contexts)
            )

    prompt = prompt + "\nReply always in a json format with the following format \n{\"name_of_mentor\":...,\"description_of_why_this_mentor_is_a_good_choice\":...,}"
    response = openai.ChatCompletion.create(model="gpt-4-1106-preview",
    messages=[{"role": "user", "content":prompt}],
    max_tokens=800)

    print(prompt)

    return response['choices'][0]['message']['content']

@app.route("/api/query", methods=["POST"])
def query():
    data = request.get_json()
    if "query" not in data:
        return jsonify({"error": "Missing 'query' field in the request data"}), 400

    input_query = data["query"]

    # 1. Embed the input query using Ada model
    embeddings = openai.Embedding.create(model="text-embedding-ada-002", input=[input_query])

    # 2. Search in Pinecone using the embeddings
    pinecone_results = search_in_pinecone(embeddings.data[0].embedding)

    # 3. Pass the retrieved result to ChatGPT as context and get a response
    if pinecone_results:
        context = [
            x['metadata']['text'] for x in pinecone_results['matches']
        ]
        response = chat_with_gpt(context,input_query)
        return jsonify({"input_query": input_query, "response": response, "context":context})

    return jsonify({"input_query": input_query, "response": "No results found in Pinecone."})

if __name__ == "__main__":
    app.run(debug=True, port=5002)