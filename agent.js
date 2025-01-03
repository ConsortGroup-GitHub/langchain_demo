import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';
import { createOpenAIFunctionsAgent, AgentExecutor } from 'langchain/agents';
import { TavilySearchResults } from '@langchain/community/tools/tavily_search';


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

// Call Agent
const response = await agentExecutor.invoke({
    input: "Quel temps fait t'il actuellement à Lille, en France ?",
});

console.log(response);
