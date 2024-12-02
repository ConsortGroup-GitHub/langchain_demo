import { ChatOpenAI } from "@langchain/openai";
import * as dotenv from 'dotenv';
dotenv.config();

const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7,
    maxTokens: 1000,
    verbose: true
});

// prompt simple avec invoke, unique :
const simpleResponse = await model.invoke("Ecris moi STP les paroles d'une chanson sur l'intelligence artificielle.");
console.log(simpleResponse);

// plusieurs prompts simultanés avec batch :
// const batchResponse = await model.batch(["Bonjour !", "Comment vas-tu aujourd'hui ?"]);
// console.log("Réponse batch : "+batchResponse.content);

// // Prompt plus complexe avec stream :

// const streamResponse = await model.stream("Ecris moi STP les paroles d'une chanson sur l'intelligence artificielle.");

// console.log("Réponse Stream :")
// for await (const chunk of streamResponse) {
//     console.log(chunk?.content);
// }
