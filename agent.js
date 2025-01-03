import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';
import { createOpenAIFunctionsAgent, AgentExecutor } from 'langchain/agents';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import { AIMessage, HumanMessage } from '@langchain/core/messages';

import readline from 'readline';


import * as dotenv from 'dotenv';
dotenv.config();

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
const tools = [searchTool];

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