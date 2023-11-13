import './Demo.css';
import { useNavigate } from 'react-router-dom';
import MovieCard from '../../src/components/MovieCard.js';

function Demo() {
    const navigate = useNavigate();

    const handleClick = () => {
        navigate('/');
    }

    return (
        <div className="main-page">
            <header className="header">
                Demo
            </header>
            <MovieCard movieId='0' />
            <button className="button" onClick={handleClick}>
                Exit
            </button>
        </div>
    );
}

export default Demo;
