import './App.css';
import { useNavigate } from 'react-router-dom';

function App() {
    const navigate = useNavigate();

    const handleClick = () => {
        navigate('/Demo');
    }

    return (
        <div className="main-page">
            <header className="header">
                Movie Recommendation
            </header>
            <header className="subheader">
                Project 4: Group 17
            </header>
            <p className="groupnames">
                <em>
                    Narasimha Arun Orunganti <br />
                    Melodi Kekskin <br />
                    Danielle Anders <br />
                    Shubh Arora <br />
                    Akshaj Kumar <br />
                    Bhanu Pratap Nayudori
                </em>
            </p>
            <button className="button" onClick={handleClick}>
                Start
            </button>
    </div>
  );
}

export default App;
