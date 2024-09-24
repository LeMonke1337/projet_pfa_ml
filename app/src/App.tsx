import './App.css'
import TabComponent from './custom-components/TabComponent'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'


const queryClient = new QueryClient() ;

function App() {

  return (
    <>
      <main className='max-w-screen-md  min-h-screen m-auto p-2 '>
        <h1 className='text-2xl font-bold mb-10'>
          Hybrid recommendation system
        </h1>
        <div>
          <div>
            <QueryClientProvider client={queryClient}>
            <TabComponent />
            </QueryClientProvider>
          </div>
        </div>
      </main>
    </>
  )
}

export default App
