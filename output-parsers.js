import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser, CommaSeparatedListOutputParser, StructuredOutputParser } from '@langchain/core/output_parsers';
import * as z from 'zod';

import * as dotenv from 'dotenv';
dotenv.config();

// Create model
const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 0.7
})

// Utilisation de StringOutputParser pour avoir un résultat sous forme de string :
async function callStringOutputParser(){

    // Create Prompt Template avec fromTemplate :
    // const prompt = ChatPromptTemplate.fromTemplate(
    //     "Tu es un humoriste. Raconte une histoire drôle basée sur le mot suivant : {input}"
    // );

    // Create Prompt Template avec fromMessages :
    const prompt = ChatPromptTemplate.fromMessages([
        ["system", "Tu es un humoriste. Raconte une histoire drôle basée sur un mot fourni par l'utilisateur"],
        ["human", "{input}"]
    ]);

    // Create Parser
    const parser = new StringOutputParser();

    // console.log(await prompt.format({input: "poulet"}));

    // Create Chain
    const chain = prompt.pipe(model).pipe(parser);

    // Call Chain
    return await chain.invoke({
        input: "bisounours"
    });
}

// Utilisation de CommaSeparatedListOutputParser pour avoir un résultat sous forme d'array :
async function callListOutputParser(){
    // Create Prompt Template avec fromTemplate :
    const prompt = ChatPromptTemplate.fromTemplate(
        "Donne moi stp 5 synonymes, séparés par des virgules, du mot suivant : {word}"
    );

    // Create Parser
    const parser = new CommaSeparatedListOutputParser();

    // Create Chain
    const chain = prompt.pipe(model).pipe(parser);

    // Call Chain
    return await chain.invoke({
        word: "hippopotame"
    });
}

// Utilisation de Structured Output Parser pour avoir un résultat sous forme de Json :
async function callStructuredParser(){
    // Create Prompt Template avec fromTemplate :
    const prompt = ChatPromptTemplate.fromTemplate(`
        Extrais les informations depuis la phrase suivante.
        Instructions de format : {format_instructions}
        Phrase : {phrase}
    `);

    // Create Parser
    const parser = StructuredOutputParser.fromNamesAndDescriptions({
        name: "le nom de la personne",
        age: "l'âge de la personne"
    });

    // Create Chain
    const chain = prompt.pipe(model).pipe(parser);

    // Call Chain
    return await chain.invoke({
        phrase: "Cela fait longtemps que Greg voulait se rendre aux US. Maintenant qu'il en est à 32 printemps, c'est chose faite !",
        format_instructions: parser.getFormatInstructions()
    });
}

// Utilisation de ZOD Output Parser pour avoir un résultat sous forme de Json, plus complexe (nécessite 'npm install zod') :
async function callZodOutputParser(){
    // Create Prompt Template avec fromTemplate :
    const prompt = ChatPromptTemplate.fromTemplate(`
        Extrais les informations depuis la phrase suivante.
        Instructions de format : {format_instructions}
        Phrase : {phrase}
    `);

    // Create Parser
    const parser = StructuredOutputParser.fromZodSchema(
        z.object({
            recipe: z.string().describe("Nom de la recette"),
            ingredients: z.array(z.string()).describe("nom d'un ingrédient"),
            alcoholPresence: z.boolean().describe("Présence ou non d'alcool dans la recette."),
            nutritionalScore: z.string().describe("Une note de nutriscore de A à F")
        })
    );

    // Create Chain
    const chain = prompt.pipe(model).pipe(parser);

    // Call Chain
    return await chain.invoke({
        //phrase: "Les ingrédients pour des spaghetti Bolognèse sont de la crème, des spaghettis, du vin, des lardons, des tomates, des oeufs et du gruyère. Sans oublier du sel et du poivre.",
        phrase: "Les ingrédients pour une salade légère sont une laitue, des carottes, une betterave rouge et des graines de courge.",
        format_instructions: parser.getFormatInstructions()
    });
}

//const response = await callStringOutputParser();
//const response = await callListOutputParser();
const response = await callStructuredParser();
// const response = await callZodOutputParser();

console.log(response);
