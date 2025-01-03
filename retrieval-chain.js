import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { CommaSeparatedListOutputParser } from '@langchain/core/output_parsers';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { createRetrievalChain } from 'langchain/chains/retrieval';

import { CheerioWebBaseLoader } from '@langchain/community/document_loaders/web/cheerio';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';



import * as dotenv from 'dotenv';
dotenv.config();

// Create model
const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7
})

// Create Prompt Template avec fromTemplate :
const prompt = ChatPromptTemplate.fromTemplate(`
    Réponds à la question de l'utilisateur stp.
    Context: {context}
    Question: {input}
`);

// Create Parser
const parser = new CommaSeparatedListOutputParser();

// Create Chain
//const chain = prompt.pipe(model).pipe(parser);
const chain = await createStuffDocumentsChain({
    llm: model,
    prompt,
    parser,
})

// Documents
// const documentA = new Document({
//     pageContent: "LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together. LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest “prompt + LLM” chain to the most complex chains (we’ve seen folks successfully run LCEL chains with 100s of steps in production)."
// });

// const documentB = new Document({
//     pageContent: "Le mot de passe est Prosper62"
// });

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

const retrievalChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever,
});


const response = await retrievalChain.invoke({
    input: "Cite STP les différents web loaders en JS pouvant être utilisés avec Langchain, en les séparant par des virgules."
    // context: [documentA, documentB]
});

console.log(response);