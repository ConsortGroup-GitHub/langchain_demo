import * as dotenv from 'dotenv';
dotenv.config();

import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from '@langchain/core/prompts';

import { ConversationChain } from 'langchain/chains';

// Memory imports
import { BufferMemory } from 'langchain/memory';
import { UpstashRedisChatMessageHistory } from '@langchain/community/stores/message/upstash_redis';


// Create model
const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.3
});

// Create Prompt Template avec fromTemplate :
const prompt = ChatPromptTemplate.fromTemplate(`
    Tu es un assistant en tant qu'intelligence artificielle.
    History: {history}
    {input}
`);

const upstashChatHistory = new UpstashRedisChatMessageHistory({
    sessionId: 'chat1',
    config: {
        url: process.env.UPSTASH_REDIS_URL,
        token: process.env.UPSTASH_REST_TOKEN,
    }
});

const memory = new BufferMemory({
    memoryKey: "history",
    chatHistory: upstashChatHistory,
});

// Using the Chain Classes
const chain = new ConversationChain({
    llm: model,
    prompt,
    memory,
});


// Create Classic Chain
// const chain = prompt.pipe(model);

// Get responses
// console.log(await memory.loadMemoryVariables());
// const input1 = {
//     input: "Mon plat préféré, ce sont les Spaghetti Bolognaise !",
// };
// const resp1 = await chain.invoke(input1);

// console.log(resp1);

console.log("Updated History: ", await memory.loadMemoryVariables());
const input2 = {
    input: "Peux-tu STP me rappeler quel est mon plat préféré STP ?",
};
const resp2 = await chain.invoke(input2);

console.log(resp2);