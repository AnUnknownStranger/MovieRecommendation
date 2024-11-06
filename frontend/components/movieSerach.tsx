// pages/index.js

import { useState, useEffect } from 'react';
import { Input } from '@/components/ui/input';

export default function MovieSearch() {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState([]);

    // Debounce hook for efficient API calls
    function useDebounce(value:any, delay:any) {
        const [debouncedValue, setDebouncedValue] = useState(value);

        useEffect(() => {
            const handler = setTimeout(() => {
                setDebouncedValue(value);
            }, delay);

            return () => {
                clearTimeout(handler);
            };
        }, [value, delay]);

        return debouncedValue;
    }

    const debouncedQuery = useDebounce(query, 300); // 300 ms debounce

    // Fetch search results from Flask API
    useEffect(() => {
        if (debouncedQuery) {
            fetch(`http://localhost:5000/search/title?q=${debouncedQuery}`)
                .then((res) => res.json())
                .then((data) => {
                    // Extract only titles
                    setResults(data.map(movie => movie.title));
                })
                .catch(error => console.error("Error fetching search results:", error));
        } else {
            setResults([]); // Clear results if query is empty
        }
    }, [debouncedQuery]);

    return (
        <div className="movie-search p-4">
            <Input
                placeholder="Search for a movie..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="w-full p-2 border rounded"
            />
            <ul className="results mt-4">
                {results.map((title, index) => (
                    <li key={index} className="text-lg font-semibold">
                        {title}
                    </li>
                ))}
            </ul>
        </div>
    );
}
