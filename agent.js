import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';
import { createOpenAIFunctionsAgent, AgentExecutor } from 'langchain/agents';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { createRetrieverTool } from 'langchain/tools/retriever';

import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

import readline from 'readline';

import * as dotenv from 'dotenv';
dotenv.config();


const loader = new CheerioWebBaseLoader("https://js.langchain.com/docs/integrations/document_loaders/web_loaders/");
const docs = await loader.load();

// 1- Découpage des documents issus du site Web en plusieurs morceaux (pour ne pas excéder le nombre de caractères possible sur un contexte)
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 20
});
const splitDocs = await splitter.splitDocuments(docs);

//2- Transformer les morceaux avec Embeddings (pour pouvoir les stocker dans une base vectorielle)
const embeddings = new OpenAIEmbeddings();

//3- Load des données dans une base vectorielle
const vectorstores = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
);

//4- Retrieve Data
const retriever = vectorstores.asRetriever({
    k: 5
});

// Create model
const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo-1106",
    temperature: 0.7
});

// Create Prompt Template avec fromMessages :
const prompt = ChatPromptTemplate.fromMessages([
    ["system", "Tu es un assistant très aidant qui s'appelle Max."],
    new MessagesPlaceholder("chat_history"),
    ["human", "{input}"],
    new MessagesPlaceholder("agent_scratchpad"),    
]);

// Create and Assign Tools
const searchTool = new TavilySearchResults();
const retrieverTool = createRetrieverTool(retriever, {
    name: "lcel_search",
    description: "Utilise cet outil lorsqu'il faut rechercher de l'information sur les web loaders.",
});
const tools = [searchTool, retrieverTool];

// Create Agent
const agent = await createOpenAIFunctionsAgent({
    llm: model,
    prompt,
    tools,
})

// Create Agent Executor
const agentExecutor = new AgentExecutor({
    agent,
    tools,
});

// Get User Input
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

const chat_history = [];

const askQuestion = () => {
    rl.question("User: ", async (input) => {

        if(input.toLowerCase() === 'exit'){
            rl.close();
            return;
        };

        // Call Agent
        const response = await agentExecutor.invoke({
            input: input,
            chat_history: chat_history,
        });
    
        console.log("Agent : ", response.output);
        chat_history.push(new HumanMessage(input));
        chat_history.push(new AIMessage(response.output));

        askQuestion();
    });
};

askQuestion();