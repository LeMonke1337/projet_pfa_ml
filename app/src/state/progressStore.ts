import { create } from "zustand"


interface RealTimeState {
    realTimeRendering: string[];
    isPending: boolean;
    setRealTimeRendering: (chunk: string) => void;
    resetRealTimeRendering: () => void;
    setIsPending: (status: boolean) => void;
    changeTab : boolean ;
    setChangeTab : (status : boolean) => void ,
    tab : string ;
    setTab : (newTab : string) => void 
}

const useProgressStore = create<RealTimeState>((set) => ({
    tab : "manual",
    setTab: (newTab : string) => set({
        tab : newTab
    }),
    changeTab : false ,
    setChangeTab : (status : boolean ) => {
        set({
            changeTab : true 
        })
    } ,
    realTimeRendering : [],
    isPending : false , 
    setRealTimeRendering : (chunk : string ) => set((state) => ({
        realTimeRendering : [...state.realTimeRendering,chunk]
    })),
    resetRealTimeRendering : () => {
        set({
            realTimeRendering : []
        })
    },
    setIsPending : (status : boolean) => {
        set({
            isPending : status ,
        })
    } 
    
}))

export default useProgressStore