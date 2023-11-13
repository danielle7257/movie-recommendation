import './MovieCard.css';
import PosterExample from '../assets/spiderman.jpg';

function MovieCard(movieId) {
	return (
		<div className="container">
			<div className="leftInnerContainer">
				<header className="title">Title</header>
				<p className="description">Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce vestibulum sem ligula, eget finibus tellus scelerisque ut. In tincidunt, ligula id faucibus mattis, diam nisi viverra est, dignissim placerat purus urna eu sapien. Aliquam finibus, metus sit amet lacinia pretium, sapien ex iaculis justo, vel elementum ipsum quam eu lectus.</p>
			</div>
			<div className="posterContainer">
				<img className="poster" src={PosterExample } />
			</div>
		</div>
	);
}

export default MovieCard;