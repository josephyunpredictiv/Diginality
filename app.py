from flask import Flask, render_template, request, jsonify
import pandas as pd, subprocess, json, string, csv, chardet, io, re

history=[]

def clean_data(input_list):
    #Clean data.
    input_list = [element.translate(str.maketrans('', '', string.punctuation)) for element in input_list]
    input_list = [element.replace('\n', '') for element in input_list]
    input_list = [element+" ? " for element in input_list]
    input_text=""
    for i in input_list:
        input_text+=i
    input_text=str(input_text)
    return input_text

def run_curl_command(curl_args):
    try:
        result = subprocess.run(curl_args, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Command execution failed: {e}")
        return None

def token_generator():
    curl_args = [
        "curl",
        "--location",
        "https://api.hyperclova.ai/v1/tokens",
        "--header",
        "Authorization: Basic PUT CREDENTIALS HERE",
    ]

    try:
        # Execute the curl command and save the output to a file
        with open('temp_token_generator.json', 'w') as output_file:
            result = run_curl_command(curl_args)
            if result:
                output_file.write(result)

        # Read the JSON data from the file and extract the access token
        result = pd.read_json('temp_token_generator.json')
        access_token = result['result']['accessToken']
        return access_token

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def segmentation(input_text, count, access_token):
    input_text = input_text.replace('\n', '')
    curl_args = [
        "curl",
        "--location",
        "https://api.hyperclova.ai/v1/api-tools/segmentation",
        "--header",
        "Content-Type: application/json",
        "--header",
        f"Authorization: Bearer {access_token}",
        "-d",
        f'{{"text": "{input_text}", "alpha": -100, "segCnt": {count}, "postProcess": false, "postProcessMaxSize": 2000, "postProcessMinSize": 500}}',
    ]

    try:
        # Execute the curl command and save the output to a file
        with open('temp_segmentation.json', 'w') as output_file:
            result = run_curl_command(curl_args)
            if result:
                output_file.write(result)

        # Read the JSON data from the file and extract the segmentation output
        result = pd.read_json('temp_segmentation.json')
        if result['status']['message'] == "OK":
            output = result['result']['topicSeg']
        else:
            output = result['status']

        return output

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def remove_strange_chars(text):
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text

def custom_tuning(training_data, access_token, epochs=3):
    curl_args = [
        "curl",
        "--location",
        "--request",
        "POST",
        "https://api.hyperclova.ai/ai-rush/v1-alpha/tasks",
        "--header",
        f"Authorization: Bearer {access_token}",
        "--header",
        "Content-Type: multipart/form-data",
        "--form",
        'name="test"',
        "--form",
        'model="LK-0"',
        "--form",
        'method="LORA"',
        "--form",
        'type="GENERATION"',
        "--form",
        'metric="PPL"',
        "--form",
        f'trainEpochs={epochs}',
        "--form",
        'learningRate="1e-4f"',
        "--form",
        f'trainingDataset=@"./{training_data}"',
    ]

    try:
        # Execute the curl command and save the output to a file
        with open('temp_custom_tuning.json', 'w') as output_file:
            result = run_curl_command(curl_args)
            if result:
                output_file.write(result)

        # Read the JSON data from the file and extract the ID and model
        result = pd.read_json('temp_custom_tuning.json')
        if result['status']['message'] == "OK":
            model = result['result']
            ID = result['result']['id']
            return ID, model
        else:
            output = result['status']
            return output

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def check_status(ID, access_token):
    curl_args = [
        "curl",
        "--location",
        "--request",
        "GET",
        f"https://api.hyperclova.ai/ai-rush/v1-alpha/tasks/{ID}",
        "--header",
        f"Authorization: Bearer {access_token}",
        "--header",
        "Content-Type: application/json",
    ]

    try:
        # Execute the curl command and save the output to a file
        with open('temp_calling_model.json', 'w') as output_file:
            result = run_curl_command(curl_args)
            if result:
                output_file.write(result)

        # Read the JSON data from the file and extract the status
        result = pd.read_json('temp_calling_model.json')
        if result['status']['message'] == "OK":
            result = result['result']['status']
            return result
        else:
            output = result['status']
            return output

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def call_tuned_model(ID, chat_history_file_path, access_token):
    status = check_status(ID, access_token)
    chat_history=json_file_to_string(chat_history_file_path)
    print(status)
    if status == 'SUCCEEDED':
        curl_args = [
            "curl",
            "--location",
            "--request",
            "POST",
            f"https://api.hyperclova.ai/ai-rush/v1-alpha/tasks/{ID}/chat-completions",
            "--header",
            f"Authorization: Bearer {access_token}",
            "--header",
            "Content-Type: application/json",
            "-d",
            f'{{"messages": {chat_history},"maxTokens": 160 }}',
        ]

        try:
            # Execute the curl command and save the output to a file
            with open('temp_chat.json', 'w') as output_file:
                result = run_curl_command(curl_args)
                print(result)
                if result:
                    output_file.write(result)

            # Read the JSON data from the file and extract the answer
            result = pd.read_json('temp_chat.json')
            if result['status']['message'] == "OK":
                role = result['result']['message']['role']
                answer = result['result']['message']['content']
                return answer
            else:
                output = result['status']
                return output
        except Exception as e:
            print(f"An error occurred: {e}")
            return 'Error'

    elif status == 'RUNNING' or status == 'WAIT':
        return 'Submit later, still training'
    else:
        return 'Error'

def call_untuned_model(chat_history_file_path, access_token):
    chat_history=json_file_to_string(chat_history_file_path)

    curl_args = [
        "curl",
        "--location",
        "--request",
        "POST",
        f"https://api.hyperclova.ai/ai-rush/v1-alpha/chat-completions/LK-0",
        "--header",
        f"Authorization: Bearer {access_token}",
        "--header",
        "Content-Type: application/json",
        "-d",
        f'{{"messages": {chat_history},"maxTokens": 160 }}',
    ]

    try:
        # Execute the curl command and save the output to a file
        with open('temp_chat.json', 'w') as output_file:
            result = run_curl_command(curl_args)
            print(result)
            if result:
                output_file.write(result)

        # Read the JSON data from the file and extract the answer
        result = pd.read_json('temp_chat.json')
        if result['status']['message'] == "OK":
            role = result['result']['message']['role']
            answer = result['result']['message']['content']
            return answer
        else:
            output = result['status']
            return output

    except Exception as e:
        print(f"An error occurred: {e}")
        return 'Error'

def json_file_to_string(input_file_path):
    with open(input_file_path, 'r') as json_file:
        data = json.load(json_file)
    json_string = json.dumps(data)

    return json_string

def chat_history_generator(data_list, start_role="user"):
    role = start_role
    chat_data = []

    for content in data_list:
        chat_data.append({"role": role, "content": content})
        role = "user" if role == "assistant" else "assistant"

    with open("chat_history.json", 'w') as json_file:
        json.dump(chat_data, json_file, indent=2)

def chatting(prompt, access_token, custom=False, ID=None):
    history.append(prompt)
    chat_history_generator(history, start_role="user")
    if custom:
        if ID != None:
            answer = call_tuned_model(ID, "chat_history.json", access_token)
        else:
            return "Need ID"
    else:
        answer=call_untuned_model("chat_history.json", access_token)
    history.append(answer)
    return answer

def generate_starter_question2(subject, access_token):
    history.clear()
    starter_question=f"유저에게 {subject}에 대해 관심이 있는지 물어봐. 첫번째 물음표 다음은 다 삭제해."
    history.append(starter_question)
    prompt = f"그 {subject}에 대해 질문을 한 번만 해봐. 첫번째 물음표 다음은 유저에게 보여주지 마."
    history.append(prompt)
    chat_history_generator(history, start_role="assistant")
    answer = call_untuned_model("chat_history.json", access_token)
    answer=cut_string(answer)
    history.append(answer)
    history.pop(0)
    history.pop(0)
    print(history)
    return answer

def generate_question(reply, access_token):
    history.append(reply)
    chat_history_generator(history, start_role="assistant")
    answer = call_untuned_model("chat_history.json", access_token)
    answer=cut_string(answer)
    history.append(answer)
    print(history)
    return answer

def generate_csv_from_list(data_list):
    history.pop(-1)
    # Divide the list into chunks of size 2 (input_text, output_text)
    input_output_pairs = [data_list[i:i+2] for i in range(0, len(data_list), 2)]
    with open("output_data.csv", 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)

        # Write the header row
        writer.writerow(["input_text", "output_text"])

        # Write the data
        writer.writerows(input_output_pairs)

def generate_json_from_list(data_list):
    history.pop(-1)
    # Divide the list into chunks of size 2 (input_text, output_text)
    input_output_pairs = [data_list[i:i+2] for i in range(0, len(data_list), 2)]

    output_data = []
    for pair in input_output_pairs:
        input_text, output_text = pair
        output_data.append({
            "input_text": input_text,
            "output_text": output_text
        })
    with open("output_data.json", 'w') as json_file:
        json.dump(output_data, json_file, indent=2)

def cut_string(input_str):
    cut_string = input_str.split('?', 1)[0]
    cut_string=cut_string+"? "
    return cut_string

def generate_starter_question(subject, access_token):
    history.clear()
    starter_question=f"You are interested in learning about the other person. Talk to the user about {subject}"
    history.append(starter_question)
    prompt = f"Now generate one question on {subject}"
    history.append(prompt)
    chat_history_generator(history, start_role="assistant")
    answer = call_untuned_model("chat_history.json", access_token)
    answer=cut_string(answer)
    history.append(answer)
    history.pop(0)
    history.pop(0)
    print(history)
    return answer

code_executed = False
indexs=0
custom_run=False
ID=""
history=[]
access_token=token_generator()


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])


def chat():
    msg = request.form["msg"]
    input = msg
    print("input:"+input)
    return get_Chat_response(input)



def get_Chat_response(text):
    global indexs
    global train
    global custom_run
    global ID
    global history
    global code_executed
    print(text=='custom')
    if code_executed==False:
        print("index is 0?")
        answer=generate_starter_question(text, access_token)
        code_executed=True
        print(history)
        return answer
    elif (text=='wonyoung'):
        if len(history)>=2:
            print("text=custom")
            generate_csv_from_list(history)
            df = pd.read_csv('output_data.csv')
            input_list=list(df["input_text"])
            count=len(input_list)
            output_list=list(df["output_text"])
            input_text=clean_data(input_list)
            C_ID=[]
            for i in range(len(df["input_text"])):
                C_ID.append(i)
            TID=0
            T_ID=[]
            T_ID.append(TID)
            if len(C_ID) != 1:
                for i in range(len(C_ID)-1):
                    print('adding onto T_ID')
                    if C_ID[i]==C_ID[i+1]:
                        TID+=1
                    else:
                        TID=0
                    T_ID.append(TID)
            print(len(C_ID))
            print(C_ID)
            print(len(T_ID))
            print(T_ID)
            final_df=pd.DataFrame({"C_ID":C_ID, "T_ID":T_ID, "Text":input_list, "Completion":output_list})
            # Column name to process
            column_name = 'Text'
            # Apply the function to the specified column
            final_df[column_name] = final_df[column_name].apply(remove_strange_chars)
            # Column name to process
            column_name = 'Completion'
            # Apply the function to the specified column
            final_df[column_name] = final_df[column_name].apply(remove_strange_chars)
            csv_content = final_df.to_csv(index=False, encoding='utf-8')
            file_path = 'train.csv'

            # Write the CSV content to the file with UTF-8 encoding
            with io.open(file_path, 'w', encoding='utf-8-sig') as file:
                file.write(csv_content)
            train=True
            ID, model = custom_tuning("train.csv", access_token, epochs=20)
            custom_run=True
            history=[]
            answer='Your Diginality is in creation. Please wait for 5 minutes.'
            return answer
        else:
            return 'please try again after entering more prompts'
    elif custom_run==True:
        print("custom")
        answer=chatting(text, access_token, custom=True, ID=ID)
        #history.append(text)
        #chat_history_generator(history, start_role="user")
        #answer = call_tuned_model(ID, "chat_history.json", access_token)
        return answer
    elif text=='default':
        print("default")
        custom_run=False
        code_executed=False
        history=[]
        answer='Default model activiated, ask question.'
        return answer
    else:
        print("else")
        answer=generate_question(text, access_token)
        return answer


if __name__ == '__main__':
    app.run()
