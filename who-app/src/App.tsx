import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import RootLayout from './layouts/RootLayout';
import HomePage from './pages/HomePage';
import LibraryPage from './pages/LibraryPage';
import PracticePage from './pages/PracticePage';
import BuildProgressPage from './pages/BuildProgressPage';
import GlobalPracticePage from './pages/GlobalPracticePage';
import BookPracticePage from './pages/BookPracticePage';
import { ImageViewer } from './components/ImageViewer';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<RootLayout />}>
            <Route path="/" element={<HomePage />} />
            <Route path="/library/:name" element={<LibraryPage />} />
            <Route path="/library/:name/practice" element={<PracticePage />} />
            <Route path="/practice/random" element={<GlobalPracticePage />} />
            <Route path="/build/:name" element={<BuildProgressPage />} />
            <Route path="/practice/books" element={<BookPracticePage />} />
            <Route path="/practice/books/:slug" element={<BookPracticePage />} />
          </Route>
        </Routes>
      </BrowserRouter>
      <ImageViewer />
    </QueryClientProvider>
  );
}
