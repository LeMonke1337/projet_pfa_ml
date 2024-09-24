import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { ManualPost } from "@/api/queries";
import { useToast } from "@/hooks/use-toast";
import useProgressStore from "@/state/progressStore"

export default function ManualForm() {
    const { toast } = useToast()
    const [experience, setExperience] = useState<string>("");
    const [education, setEducation] = useState<string>("");
    const [skills, setSkills] = useState<string>("");
    
    const { setRealTimeRendering , setIsPending , setTab , setChangeTab } = useProgressStore()

    const { mutate, isPending , isSuccess} = useMutation({
        mutationFn: (data: { experience: string, education: string, skills: string }) => {
            return ManualPost(data, (chunk) => {
                setRealTimeRendering(chunk)
            })
        },
        onMutate : () => {
            setIsPending(true)
            setChangeTab(true)
            setTab("result")
        },
        onError: () => {
            setIsPending(false)
        },
        onSuccess : () => {
            setIsPending(false)
        }
    })

    const sendData = () => {
        console.log(education, experience, skills)

        mutate({ experience: experience, education: education, skills: skills })

    }


    return (
        <div>
            <div>
                <Label>
                    Experience
                </Label>
                <Textarea disabled={isPending}  value={experience} onChange={(e) => setExperience(e.target.value)} placeholder="From work to internships count as experience. " >
               

                </Textarea>
            </div>
            <div>
                <Label>
                    Education
                </Label>
                <Textarea disabled={isPending} value={education} onChange={(e) => setEducation(e.target.value)} placeholder="Your university track suffice. " >
                </Textarea>
            </div>
            <div>
                <Label>
                    Skills
                </Label>
                <Textarea disabled={isPending} value={skills} onChange={(e) => setSkills(e.target.value)} placeholder="Your skills delimited by comma ">

                </Textarea>
            </div>
            <div className="mt-2 w-full">
                <Button

                    disabled={!(experience && education && skills) || isPending}
                    onClick={sendData} className="w-full">
                    {isPending ? "loading.." : "submit"}
                </Button>
            </div>
            
               

        </div>
    )
}