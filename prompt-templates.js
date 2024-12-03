import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from '@langchain/core/prompts';

import * as dotenv from 'dotenv';
dotenv.config();

// Create model
const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7
});

// Create Prompt Template avec fromTemplate :
// const prompt = ChatPromptTemplate.fromTemplate(
//     "Tu es un humoriste. Raconte une histoire drôle basée sur le mot suivant : {input}"
// );

// Create Prompt Template avec fromMessages :
const prompt = ChatPromptTemplate.fromMessages([
    ["system", "Tu es un humoriste. Raconte une histoire drôle basée sur un mot fourni par l'utilisateur"],
    ["human", "{input}"]
]);


// console.log(await prompt.format({input: "poulet"}));

// Create Chain
const chain = prompt.pipe(model);

// Call Chain
const response = await chain.invoke({
    input: "chat"
});

console.log(response);

