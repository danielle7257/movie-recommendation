import './MovieCard.css';
import PosterExample from '../assets/spiderman.jpg';

function MovieCard({ name, imgUrl }) {

	return (
		<div className="container">
			<div className="posterContainer">
				<img className="poster" src={imgUrl} alt=" " />
			</div>
			<div className="leftInnerContainer">
				<header className="title">{name}</header>
			</div>
		</div>
	);
}

export default MovieCard;