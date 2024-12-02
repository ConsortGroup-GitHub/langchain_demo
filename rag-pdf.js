import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { CommaSeparatedListOutputParser } from "@langchain/core/output_parsers";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";
import * as dotenv from "dotenv";
dotenv.config();

// Create model
const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7
});

// Create Prompt Template avec fromTemplate
const prompt = ChatPromptTemplate.fromTemplate(`
    Réponds à la question de l'utilisateur stp.
    Context: {context}
    Question: {input}
`);

// Create Parser
const parser = new CommaSeparatedListOutputParser();

// Create Chain
const chain = await createStuffDocumentsChain({
    llm: model,
    prompt,
    parser,
});

// Load PDF Document using PDFLoader
const loader = new PDFLoader("data/example.pdf");
const docs = await loader.load();

// 1- Split PDF documents into multiple chunks (to avoid exceeding context size limits)
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 20
});
const splitDocs = await splitter.splitDocuments(docs);

// 2- Transform the chunks with Embeddings (for storing in a vector database)
const embeddings = new OpenAIEmbeddings();

// 3- Load data into a vector store
const vectorstores = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
);

// 4- Retrieve Data
const retriever = vectorstores.asRetriever({
    k: 20
});

const retrievalChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever,
});

const response = await retrievalChain.invoke({
    input: "Quelles sont les compétences techniques demandées dans cet appel d'offres ?"
});

console.log(response);
