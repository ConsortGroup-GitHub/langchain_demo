import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";

import { MultiFileLoader } from "langchain/document_loaders/fs/multi_file";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { DocxLoader } from "@langchain/community/document_loaders/fs/docx";
import { PPTXLoader } from "@langchain/community/document_loaders/fs/pptx";
import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";

import * as dotenv from "dotenv";
dotenv.config();

// Create model
const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.1
});

// Create Prompt Template avec fromTemplate
const prompt = ChatPromptTemplate.fromTemplate(`
    Réponds à la question de l'utilisateur stp.
    Context: {context}
    Question: {input}
`);

// Create Parser
const parser = new StringOutputParser();

// Create Chain
const chain = await createStuffDocumentsChain({
    llm: model,
    prompt,
    parser,
});

// Load All files
const loader = new MultiFileLoader(
    [
      "data/Example.csv",
      "data/Example.docx",
      "data/example.pdf",
      "data/Example.pptx",
    ],
    {
      ".csv": (path) => new CSVLoader(path),
      ".docx": (path) => new DocxLoader(path),
      ".pdf": (path) => new PDFLoader(path),
      ".pptx": (path) => new PPTXLoader(path),
    }
  );
const docs = await loader.load();

// 1- Split Multiple documents into multiple chunks (to avoid exceeding context size limits)
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
    //input: "Quel est le taux de certifiés ISTQB Fondation en 2019 ?"
    //input: "Quelles sont les personnes qui génèrent des Quota Amounts ?"
    //input: "C'est le CV de qui ? Quelles sont les compétences qu'on y trouve ?"
    input: "Est-ce-que Lies peut coller au besoin Decathlon ? Pourrais-tu donner une note sur 100 de complétude de son profil par rapport à cette mission ?"
});

console.log(response);
