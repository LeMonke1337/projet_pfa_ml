import { Tabs, TabsList, TabsTrigger , TabsContent } from "@/components/ui/tabs";
import ManualForm from "./ManualForm";
import LlmForm from "./LlmForm";
import { useEffect, useState } from "react";
import ResultTab from "./ResultTab";
import useProgressStore from "@/state/progressStore";

export default function TabComponent(){

    const {setTab , tab , changeTab } = useProgressStore()
    
    
    

    return (
        <div>
            <Tabs value={tab} onValueChange={(value) => {
                setTab(value)
                }}>
                <TabsList >
                    <TabsTrigger  value="manual">
                        Manual
                    </TabsTrigger>
                    <TabsTrigger value="llm">
                        LLM usage
                    </TabsTrigger>
                    <TabsTrigger className={changeTab ?  "" : "hidden" } value="result">
                        hidden 
                    </TabsTrigger>
                </TabsList>
                <TabsContent forceMount  value="manual" hidden={tab !== "manual"}>
                    <ManualForm />
                </TabsContent>
                <TabsContent forceMount hidden={tab !== "llm" } value="llm">
                    <LlmForm /> 
                </TabsContent>
                <TabsContent forceMount value="result"  hidden={tab !== "result"}>
                    <ResultTab />
                </TabsContent>
            </Tabs>

        </div>
    )
}