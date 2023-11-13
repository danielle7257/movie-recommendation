import './SignUp.css';
import { useNavigate } from 'react-router-dom';

function SignUp() {
    const navigate = useNavigate();

    const handleClick = () => {
        navigate('/Demo');
    }

    return (
        <div className="main-page">
            <header className="header">
                Sign Up
            </header>
            <button className="button" onClick={handleClick }>
                Continue
            </button>
        </div>
    );
}

export default SignUp;
