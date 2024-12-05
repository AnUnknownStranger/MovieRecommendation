'use client';
import { useState, useEffect } from 'react';
import { Input } from '@/components/ui/input';
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog";

interface Movie {
    id: string;
    title: string;
}

interface MovieDetails {
    id: string;
    title: string;
    overview: string;
    releaseDate: string;
    runtime: string;
    genres: string[];
    productionCompanies: string[];
    popularity: string;
    voteAverage: string;
    voteCount: string;
}

export default function MovieSearch() {
    const [query, setQuery] = useState('');
    const [results, setResults] = useState<Movie[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [selectedMovie, setSelectedMovie] = useState<MovieDetails | null>(null);
    const [isDialogOpen, setIsDialogOpen] = useState(false);
    const [recommendations, setRecommendations] = useState<string[]>([]);
    const [showRecommendations, setShowRecommendations] = useState(false);

    function useDebounce(value: string, delay: number) {
        const [debouncedValue, setDebouncedValue] = useState(value);
        useEffect(() => {
            const handler = setTimeout(() => {
                setDebouncedValue(value);
            }, delay);
            return () => clearTimeout(handler);
        }, [value, delay]);
        return debouncedValue;
    }

    const debouncedQuery = useDebounce(query, 300);

    useEffect(() => {
        if (debouncedQuery) {
            fetch(`http://localhost:8080/search/title?q=${debouncedQuery}`)
                .then((res) => res.json())
                .then((data) => {
                    setResults(data.map((movie: { id: string; title: string; }) => ({
                        id: movie.id,
                        title: movie.title,
                    })));
                })
                .catch(error => {
                    console.error("Error fetching search results:", error);
                    setError("Failed to fetch search results");
                });
        } else {
            setResults([]);
        }
    }, [debouncedQuery]);

    const handleLike = async (movieId: string) => {
        try {
            const response = await fetch('http://localhost:8080/api/like', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ movieId }),
            });
            if (!response.ok) throw new Error('Failed to like movie');
            alert('Movie liked successfully!');
        } catch (error) {
            console.error("Error liking movie:", error);
            setError("Failed to like movie");
        }
    };

    const handleDislike = async (movieId: string) => {
        try {
            const response = await fetch('http://localhost:8080/api/dislike', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ movieId }),
            });
            if (!response.ok) throw new Error('Failed to dislike movie');
            alert('Movie disliked successfully!');
        } catch (error) {
            console.error("Error disliking movie:", error);
            setError("Failed to dislike movie");
        }
    };

    const MakeRecommendation = async (movieTitle: string) => {
        try {
            const response = await fetch('http://localhost:8080/api/MakeRecommendation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                credentials: 'include',
                body: JSON.stringify({ movieTitle }),
            });
            if (!response.ok) throw new Error('Failed to Make Recommendation');
            const data = await response.json();
            setRecommendations(data.Recommendations);
            setIsDialogOpen(true);
            setShowRecommendations(true);
        } catch (error) {
            console.error("Error Making Recommendation:", error);
            setError("Failed to Make Recommendation");
        }
    };

    const fetchMovieDetails = async (movieId: string) => {
        try {
            const response = await fetch(`http://localhost:8080/api/movie/details?id=${movieId}`);
            if (!response.ok) throw new Error("Failed to fetch movie details");
            const data = await response.json();
            setSelectedMovie(data);
            setIsDialogOpen(true);
            setShowRecommendations(false);
        } catch (error) {
            console.error("Error fetching movie details:", error);
            setError("Failed to fetch movie details");
        }
    };

    
    return (
        <div className="movie-search p-8 space-y-4">
            {error && <p className="text-red-500">{error}</p>}
            <Input
                placeholder="Search for a movie..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                className="w-full p-2 border rounded"
            />
            <ul className="results mt-4 space-y-2">
                {results.map((movie) => (
                    <li key={movie.id} className="text-lg font-semibold flex items-center justify-between p-2 border rounded-lg">
                        <span
                            className="cursor-pointer text-blue-500"
                            onClick={() => fetchMovieDetails(movie.id)}
                        >
                            {movie.title}
                        </span>
                        <div className="space-x-2">
                        <button
                                className="px-4 py-2 bg-blue-800 text-white rounded"
                                onClick={() => MakeRecommendation(movie.title)}
                            >
                                Find Similar
                            </button>
                            
                            <button
                                className="px-4 py-2 bg-green-500 text-white rounded"
                                onClick={() => handleLike(movie.id)}
                            >
                                Like
                            </button>
                            <button
                                className="px-4 py-2 bg-red-500 text-white rounded"
                                onClick={() => handleDislike(movie.id)}
                            >
                                Dislike
                            </button>
                        </div>
                    </li>
                ))}
            </ul>

            {selectedMovie && (
                <Dialog open={isDialogOpen && !showRecommendations} onOpenChange={setIsDialogOpen}>
                    <DialogTrigger asChild>
                        <button className="hidden">Open</button>
                    </DialogTrigger>
                    <DialogContent>
                        <DialogHeader>
                            <DialogTitle>{selectedMovie.title}</DialogTitle>
                            <DialogDescription>
                                <p><strong>Overview:</strong> {selectedMovie.overview}</p>
                                <p><strong>Release Date:</strong> {selectedMovie.releaseDate}</p>
                                <p><strong>Runtime:</strong> {selectedMovie.runtime} mins</p>
                                <p><strong>Genres:</strong> {selectedMovie.genres.join(", ")}</p>
                                <p><strong>Production Companies:</strong> {selectedMovie.productionCompanies.join(", ")}</p>
                                <p><strong>Popularity:</strong> {selectedMovie.popularity}</p>
                                <p><strong>Average Vote:</strong> {selectedMovie.voteAverage}</p>
                                <p><strong>Vote Count:</strong> {selectedMovie.voteCount}</p>
                            </DialogDescription>
                        </DialogHeader>
                    </DialogContent>
                </Dialog>
            )}

            {recommendations &&(
                <Dialog open={isDialogOpen && showRecommendations} onOpenChange={setIsDialogOpen}>
                    <DialogTrigger asChild>
                        <button className="hidden">Open</button>
                    </DialogTrigger>
                    <DialogContent>
                        <DialogHeader>
                            <DialogTitle>{"Recommendations"}</DialogTitle>
                            <DialogDescription>
                                <div>
                                    <ul>
                                        {recommendations.map((rec, index) => (
                                            <li key={index}>{rec}</li>
                                        ))}
                                    </ul>
                                </div>
                            </DialogDescription>
                        </DialogHeader>
                    </DialogContent>
                </Dialog>
            )}

        </div>
    );
}
