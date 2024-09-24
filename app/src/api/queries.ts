import { error } from "console";


const url = "http://localhost:8000"

export const ManualPost = async (data: { experience: string; education: string; skills: string } , onProgress : (chunk : string) => void ) => {


    try {

        const response = await fetch(`${url}/model`, {
            headers: {
                "Content-Type": "application/json"
            },
            method: "POST",
            body: JSON.stringify(data)
        })

        const reader = response.body?.getReader()
        const decoder = new TextDecoder()
        let finished = false

        while (!finished) {
            const { value, done: streamDone } = await reader!.read();
            finished = streamDone
            if (value) {
                const chunk = decoder.decode(value, { stream: true });
                onProgress(chunk)
            }
        }

        if (!response.ok) {
            const errorData = await response.json();
            throw errorData;;
        }

        const dataOutput = await response.json();


        console.log(dataOutput);

        return dataOutput;

    } catch (e) {
        throw e;
    }
}