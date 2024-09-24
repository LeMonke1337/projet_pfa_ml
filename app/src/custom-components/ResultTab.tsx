import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import useProgressStore from "@/state/progressStore";
import { BookmarkCheck, Check, CheckCheck, CircleX, Loader2, PartyPopper, Terminal } from "lucide-react";
import { useState, useEffect, useRef } from "react";
import ReactMarkdown from 'react-markdown'
import { text } from "stream/consumers";
import { Pie } from "@visx/shape";
import { Group } from "@visx/group";
import { scaleOrdinal } from "d3-scale";
import { schemeCategory10 } from "d3-scale-chromatic";

type ChartData = {
    label: string;
    value: number;
};


const width = 400;
const height = 400;
const radius = Math.min(width, height) / 2;

const colorScale = scaleOrdinal(schemeCategory10);


export default function ResultTab() {

    const [currentIndex, setCurrentIndex] = useState(-1);
    const [chartData, setChartData] = useState<ChartData[]>([]);
    const { realTimeRendering, isPending } = useProgressStore()
    const [isChartDataComplete, setIsChartDataComplete] = useState(false); // Track whether chartData is complete

    const updateChartData = (newData: string) => {
        const [category, percentage] = newData.split(": ");
        const percentValue = parseFloat(percentage.replace("%", ""));


        setChartData((prevData) => [
            ...prevData,
            { label: category, value: percentValue },
        ]);
    };

    useEffect(() => {
        realTimeRendering.forEach((chunk) => {
            if (chunk.startsWith("[DATA]")) {
                
                const text = chunk.replace("[DATA]", "");
                console.log(chartData.length)



                updateChartData(text);
                if (chunk.startsWith("[FINISHED]")) {
                    console.log(chartData)
                    setIsChartDataComplete(true);
                }
            }
        });

        if (realTimeRendering.length > 0) {
            
            setCurrentIndex(realTimeRendering.length - 1);  // Always point to the latest chunk
        }
    }, [realTimeRendering, chartData , isChartDataComplete]);

    return (
        <div className="space-y-2 ">

        

            <div className={`mt-4 space-y-2 `}>
                <div>
                    <Alert >
                        {isPending ?
                            <Loader2 className="w-4 h-4 animate-spin" />
                            :
                            <BookmarkCheck className="w-4 h-4" />
                        }
                        <AlertTitle>
                            {isPending ? "Data is being processed right now...." : "Data processing finished."}
                        </AlertTitle>
                    </Alert>
                </div>
                <div className="space-y-2" >
                    {realTimeRendering && realTimeRendering.map((chunk, index) => {

                        if (chunk.startsWith("[SUCCESS]")) {
                            const text = chunk.replace("[SUCCESS]", "")
                            return (
                                <Alert key={index}>

                                    <PartyPopper className={`w-4 h-4  `} />



                                    <AlertTitle>
                                        {text}
                                    </AlertTitle>
                                </Alert>
                            )

                        } else if (chunk.startsWith("[CHECK]")) {
                            const text = chunk.replace("[CHECK]", "")
                            return (
                                <Alert key={index}>
                                    {index == currentIndex ?
                                        <Loader2 className={`w-4 h-4 animate-spin`} />
                                        :
                                        <CheckCheck className="w-4 h-4" />
                                    }
                                    <AlertTitle>
                                        {text}
                                    </AlertTitle>
                                </Alert>
                            )
                        } else if (chunk.startsWith("[ERROR]")) {
                            const text = chunk.replace("[ERROR]", "")
                            return (
                                <Alert variant={"destructive"}>
                                    <CircleX className="w-4 h-4" />
                                    <AlertTitle>
                                        {text}
                                    </AlertTitle>
                                </Alert>
                            )
                        } else if (chunk.startsWith("[CARD]")) {
                            const text = chunk.replace("[CARD]", "")
                            return (
                                <Card>
                                    <CardHeader>
                                        <CardDescription>
                                            <ReactMarkdown>
                                                {text}
                                            </ReactMarkdown>
                                        </CardDescription>
                                    </CardHeader>
                                </Card>
                            )
                        } else if (chunk.startsWith("[TITLE]")) {
                            const text = chunk.replace("[TITLE]", "")
                            return (
                                <div>
                                    <Alert>
                                        <AlertTitle className="text-center">
                                            {text}
                                        </AlertTitle>
                                    </Alert>
                                </div>
                            )
                        } else if (chunk.startsWith("[FINISH]")) {
                            const text = chunk.replace("[FINISH]", "")
                            return (
                                <Alert className="border-green-600 text-green-600">
                                    <CircleX className="w-4 h-4" />
                                    <AlertTitle>
                                        {text}
                                    </AlertTitle>
                                </Alert>
                            )
                        } else if (chunk.startsWith("[START]")) {
                            const text = chunk.replace("[START]", "")
                            return (
                                <Alert className="border-green-600 text-green-600">

                                    <AlertTitle>
                                        <div dangerouslySetInnerHTML={{ __html: text }} />
                                    </AlertTitle>
                                </Alert>
                            )
                        } else if (chunk.startsWith("[CARD2]")) {
                            const text = chunk.replace("[CARD2]", "")
                            return (
                                <div key={index} className="">
                                    <Card>
                                        <CardHeader>
                                            <div>
                                                {text}
                                            </div>
                                        </CardHeader>
                                    </Card>
                                </div>
                            )

                        }

                        {
                            isChartDataComplete ?  (


                                <div id="graph">

                                    <svg width={width} height={height}>
                                        <Group top={height / 2} left={width / 2}>
                                            <Pie
                                                data={chartData}
                                                pieValue={(d) => d.value}
                                                outerRadius={radius}
                                                innerRadius={radius - 80}
                                                padAngle={0.02}
                                            >
                                                {(pie) =>
                                                    pie.arcs.map((arc, index) => (
                                                        <g key={`arc-${index}`}>
                                                            <path d={pie.path(arc)} fill={colorScale(index)} />
                                                            <text
                                                                transform={`translate(${pie.path.centroid(arc)})`}
                                                                dy=".35em"
                                                                fontSize={10}
                                                                textAnchor="middle"
                                                                fill="#fff"
                                                            >
                                                                {arc.data.label}
                                                            </text>
                                                        </g>
                                                    ))
                                                }
                                            </Pie>
                                        </Group>
                                    </svg>
                                </div>






                            ): (<div> Loading graph data...</div>)
                        }



                    })}
                </div>
            </div>

        </div>
    )
}