import './Demo.css';
import { useNavigate } from 'react-router-dom';
import React, { useState } from 'react';
import MovieCard from '../../src/components/MovieCard.js';
import RecommendedFilms from '../../src/components/RecommendedFilms.js'

function Demo() {
    const navigate = useNavigate();

    const [count, setCount] = useState(0);
    const [movieTitle, setMovieTitle] = useState("");
    const [films, setFilms] = useState([]);
    const [imageUrl, setImageUrl] = useState("");

    const movie = {
        title: "",
        url: ""
    }

    const handleClick = () => {
        navigate('/');
    }

    const handleInput = () => {
        setMovieTitle(document.getElementById("titleEntry").value);
    }

    const handleEnter = () => {
        // clear content
        setFilms([]);

        console.log("Debug 1:");
        console.log(movieTitle);

        fetch('/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 'value': movieTitle })
        })
            .then(response => response.json())
            .then((json) => {
                const baseUrl = "https://image.tmdb.org/t/p/original";
                json.forEach((item) => {
                    const movie = {
                        title: item['title'],
                        url: baseUrl.concat(item['poster_path'])
                    };
                    console.log("DEBUG 2:::::::: ");
                    console.log(movie);
                    setFilms((prevFilms) => [
                        ...prevFilms,
                        movie
                    ]);
                    console.log(films);
                })
            })
            .catch(error => {
                console.error('Error:', error);
            });

        // update film recommendation #
        setCount(films.length);
        if (count === 0) {
            document.getElementById("subheader0").textContent = "";
        }
        else {
            console.log(films);
            document.getElementById("subheader0").textContent = "Recommendations";
        }
    }

    return (
        <div className="main-page">
            <header className="header">
                Movie Recommendation Demo
            </header>
            <div className="input-container">
                <input className="input" type="text" id="titleEntry" placeholder="Enter movie title" onChange={handleInput} />
                <button className="button" onClick={handleEnter}>
                    Enter
                </button>
            </div>
            {/* <MovieCard name={movieTitle} imgUrl={imageUrl} /> */}
            <header id="subheader0" className="subheader">
                
            </header>
            <RecommendedFilms titles={films} />
            <button className="button" onClick={handleClick}>
                Exit
            </button>
        </div>
    );
}

export default Demo;
