import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import modelList from "@/llmModels.json"
import { useRef, useState } from "react";

export default function LlmForm() {

    



    const pdfRef = useRef<HTMLInputElement>(null);
    const [fileHide , setFileStatus ] = useState(false) 
    const [fileUrl, setFileUrl] = useState<string | null >(null);


    const handleFileChange = () => {
        if (pdfRef.current && pdfRef.current.files && pdfRef.current.files.length > 0) {
            const file = pdfRef.current.files[0];
            const fileUrl = URL.createObjectURL(file);
            setFileUrl(fileUrl);
        }
    };

    const hideFile = () => {
        setFileStatus(!fileHide)
    }   
    const removeFile = () => {
        setFileUrl(null)
        pdfRef.current!.value = ''
    }


    return (
        <div className="space-y-2">
            <div>
                <Label>
                    Your resume
                </Label>
                <Input onChange={handleFileChange} ref={pdfRef} type="file" accept=".pdf" />
            </div>
            <div className="space-y-2">
                <div  className={`${fileHide ? "hidden" : ""}`}>
                    <embed  src={fileUrl ?? undefined} className="w-full border " height="375" />
                </div>
                <div className={`grid grid-cols-2 gap-4  ${fileUrl == null ? "hidden" : ""}` }>
                    <Button onClick={hideFile}>
                        Hide
                    </Button>
                    <Button onClick={removeFile}>
                        Remove
                    </Button>
                </div>
            </div>
            
            <div className="mt-2 w-full">
                <Button  className="w-full">
                    submit
                </Button>
            </div>
        </div>
    )
}