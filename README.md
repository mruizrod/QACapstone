<p align="center">
    <img src="https://www.svgrepo.com/show/375425/dialogflow-cx.svg" align="center" width="30%">
</p>
<p align="center"><h1 align="center">QA CAPSTONE</h1></p>
<p align="center">
	<em>Empowering insights through seamless document interactions.</em>
</p>
<br>

##  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Project Structure](#-project-structure)
  - [ Project Index](#-project-index)
- [ Getting Started](#-getting-started)
  - [ Installation](#-installation)
  - [ Usage](#-usage)



##  Overview

QACapstone.git is a project that streamlines document parsing and query response generation for enhanced user interactions. By leveraging advanced NLP techniques and LLMs, it enables seamless extraction of insights from PDF files, empowering users to ask questions and receive accurate responses in real-time. Ideal for teams seeking efficient data analysis and interactive communication.


##  Features

|      | Feature         | Summary       |
| :--- | :---:           | ----          |
| ⚙️  | **Architecture**  | <ul><li>Utilizes **Llama2** and other NLP approaches for real-time chatbot interactions with financial reports</li><li>Configurable pipeline with PDF parser and model selection</li><li>Demonstrates efficient data ingestion and parsing capabilities</li><li>Uses **Streamlit** for interactive web application</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Central components like **llama2.py** ensure flexible instantiation of LLM/NLP models based on model type</li><li>Main conponent **pipeline.py** that integrates the usage of different parsers/models</li><li>Efficient data processing with `PDFParser` class within the project architecture</li></ul> |


##  Project Structure

```
└── QACapstone.git/
    ├── README.md
    ├── data
    │   ├── pdf
    │   └── pkl
    ├── src
    │   ├── llama2.py
    │   ├── llama2_general.py
    │   ├── nlp_base.py
    │   ├── nlp_langchain.py
    │   ├── open_ai.py
    │   ├── pipeline.py
    │   └── utils.py
	├── test
	│   └── ... (notebooks and results for experimental purposes)
    ├── main.py
    └── requirements.txt
```


###  Project Index
<details open>
	<summary><b><code>QACAPSTONE.GIT/</code></b></summary>
	<details> <!-- __root__ Submodule -->
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/mruizrod/QACapstone.git/blob/master/requirements.txt'>requirements.txt</a></b></td>
				<td>- Facilitates project dependencies management by specifying required packages and versions in the 'requirements.txt' file</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/mruizrod/QACapstone.git/blob/master/main.py'>main.py</a></b></td>
				<td>- Enables real-time chatbot interactions for financial reports, utilizing a configurable pipeline with PDF reader and model selection<br>- Users can ask questions, receive responses, and view chat history<br>- Log information is displayed in terminal for future reference.</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details> <!-- src Submodule -->
		<summary><b>src</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/mruizrod/QACapstone.git/blob/master/src/utils.py'>utils.py</a></b></td>
				<td>- Defines a guide function to transform raw user input into a guided input that contains clearer instructions for the model<br>- Defines a PDFParser class to process PDF files based on specified methods like 'unstructured' or 'pdfplumber'<br>- Once a PDF file is parsed, it'll be stored in a pickle file to boost future loading efficiency</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/mruizrod/QACapstone.git/blob/master/src/nlp_langchain.py'>nlp_langchain.py</a></b></td>
				<td>- Defines the NLP_langchain class, an NLP-based text extractor module for document parsing, embedding, and query response generation<br>- Unlike LLM, this approach directly return the extracted text from the document as its response<br>- Allows for much more efficient query respond generation</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/mruizrod/QACapstone.git/blob/master/src/nlp_base.py'>nlp_base.py</a></b></td>
				<td>- Contains the naive version of NLP_langcain, currently unused</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/mruizrod/QACapstone.git/blob/master/src/llama2.py'>llama2.py</a></b></td>
				<td>- Implements a Llama2 class for document indexing and querying with Ollama<br>- Loads and processes documents, trains the Llama-index with a specified model, and provides methods for embedding queries and generating responses</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/mruizrod/QACapstone.git/blob/master/src/llama2_general.py'>llama2_general.py</a></b></td>
				<td>- Defines a generalized Llama2 class for document indexing and querying<br>- Removes the necessity of input data and allows the chatbot to answer genaral questions<br>- Currently mainly used as the prompt generator for prompt engineering</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/mruizrod/QACapstone.git/blob/master/src/open_ai.py'>open_ai.py</a></b></td>
				<td>- Implements a class to process queries with OpenAI API, currently unused</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/mruizrod/QACapstone.git/blob/master/src/pipeline.py'>pipeline.py</a></b></td>
				<td>- Defines a class to instantiate different parsers and models based on the specified model type<br>- The function creates instances of Llama2, OpenAI, NLP_base, or NLP_langchain classes depending on the model parameter provided<br>- This allows for flexible integration of various NLP models into the pipeline for processing data</td>
			</tr>
			</table>
		</blockquote>
	</details>
</details>


##  Getting Started

###  Installation

1. Clone the QACapstone.git repository:
	```sh
	git clone https://github.com/mruizrod/QACapstone.git
	```

2. Navigate to the project directory:
	```sh
	cd QACapstone.git
	```

3. Install the project dependencies:
	```sh
	pip install -r requirements.txt
	```




###  Usage
Create a `.env` file in the root directory of the project and populate it with your API key. The file should look like this:
```
LLAMA_CLOUD_API_KEY=...
OPENAI_API_KEY=...
```
Make sure Ollama is running on your device.

Run QACapstone.git using the following command:
**Using `streamlit`** &nbsp; 
```sh
streamlit run main.py
```
