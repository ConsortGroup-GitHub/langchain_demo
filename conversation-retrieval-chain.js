import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { CommaSeparatedListOutputParser } from '@langchain/core/output_parsers';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
// import { Document } from '@langchain/core/documents';
import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';

import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { createRetrievalChain } from 'langchain/chains/retrieval';

import { MessagesPlaceholder } from '@langchain/core/prompts';
import { AIMessage, HumanMessage } from '@langchain/core/messages';
import { createHistoryAwareRetriever } from 'langchain/chains/history_aware_retriever';


import * as dotenv from 'dotenv';
dotenv.config();


// Load Data and create Vector Store
const createVectorStore = async () => {

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
    const vectorStore = await MemoryVectorStore.fromDocuments(
        splitDocs,
        embeddings
    );

    return vectorStore;
}

// Create Retrieval Chain
const createChain = async () => {

    // Create model
    const model = new ChatOpenAI({
        modelName: "gpt-3.5-turbo",
        temperature: 0.7
    });

    // Create Prompt Template avec fromTemplate :
    // const prompt = ChatPromptTemplate.fromTemplate(`
    //     Réponds à la question de l'utilisateur stp.
    //     Context: {context}
    //     Chat History: {chat_history}
    //     Question: {input}
    // `);

    // Create Prompt Template avec fromMessages :
    const prompt = ChatPromptTemplate.fromMessages([
        ["system", "Réponds à la question de l'utilisateur stp, en te basant sur le contexte suivante : {context}."],
        new MessagesPlaceholder("chat_history"),
        ["user", "{input}"],
    ]);


    // Create Parser
    const parser = new CommaSeparatedListOutputParser();

    // Create Chain
    const chain = await createStuffDocumentsChain({
        llm: model,
        prompt,
        parser,
    })

    //4- Retrieve Data
    const retriever = vectorStore.asRetriever({
        k: 5
    });

    const retrieverPrompt = ChatPromptTemplate.fromMessages([
        new MessagesPlaceholder("chat_history"),
        ["user", "{input}"],
        ["user", "Sur la base de la conversation précédente, génère une requête de recherche pour avoir l'information la plus adéquate pour la conversation"]
    ]);

    const history_aware_retriever = await createHistoryAwareRetriever({
        llm: model,
        retriever,
        rephrasePrompt: retrieverPrompt,
    });

    const conversationChain = await createRetrievalChain({
        combineDocsChain: chain,
        retriever: history_aware_retriever,
    });

    return conversationChain;
};

const vectorStore = await createVectorStore();
const chain = await createChain(vectorStore);

// Chat History
const chatHistory = [
    new HumanMessage("Bonjour !"),
    new AIMessage("Bonjour, comment puis-je t'aider aujourd'hui ?"),
    new HumanMessage("Mon nom est David."),
    new AIMessage("Bonjour David, comment puis-je t'aider aujourd'hui ?"),
    new HumanMessage("Cite STP les différents web loaders en JS pouvant être utilisés avec Langchain, en les séparant par des virgules."),
    new AIMessage("Les différents web loaders en JS pouvant être utilisés avec Langchain sont Playwright, AirtableLoader, PDFLoader, TextLoader.")
];


const response = await chain.invoke({
    input: "Tu peux me rappeler les différents Web Loaders, stp ?",
    chat_history: chatHistory
});

console.log(response);