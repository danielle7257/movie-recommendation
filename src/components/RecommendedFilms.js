import './RecommendedFilms.css';
import PosterExample from '../assets/spiderman.jpg';

function RecommendedFilms({ titles }) {
	const films = titles.map(title => (
		<div>
			<img className="recommendedPoster" src={title.url} alt="movie-poster" />
			<p>{title.title}</p>
		</div>
	));

	return (
		<div className="recommendedContainer">
			{films}
		</div>
	);
}

export default RecommendedFilms;