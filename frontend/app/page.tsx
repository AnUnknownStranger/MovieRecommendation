"use client"

import { Button } from '@/components/ui/button';
import { useState } from 'react';

export default function Home() {
  // State to store fetched data
  const [data, setData] = useState(null);

  // Fetch data from Flask server
  const handleClick = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/data');
      const result = await response.json();
      console.log("Fetched data:", result);
      setData(result);
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };

  return (
    <div>
      <Button onClick={handleClick}>Click me</Button>
      {data && (
        <div>
          <h2>Fetched Data:</h2>
          <pre>{JSON.stringify(data, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
