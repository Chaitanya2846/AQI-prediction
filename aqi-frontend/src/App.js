import React, { useState } from 'react';

function App() {
  const [formData, setFormData] = useState({
    City: 'Ahmedabad', PM2_5: 45.0, PM10: 90.0, 
    NO2: 15.0, SO2: 20.0, CO: 1.0, O3: 50.0,
  });
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');

  // Replace with the exact cities present in your dataset
  const cities = ['Ahmedabad', 'Delhi', 'Mumbai', 'Pune', 'Bengaluru'];

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: name === 'City' ? value : parseFloat(value) });
  };

  const getPrediction = async (e) => {
    e.preventDefault();
    setError('');
    const date = new Date();
    
    try {
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...formData,
          Year: date.getFullYear(),
          Month: date.getMonth() + 1,
          Day: date.getDate()
        }),
      });

      if (!res.ok) throw new Error('Backend connection failed.');
      setPrediction(await res.json());
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div style={{ maxWidth: '600px', margin: '2rem auto', fontFamily: 'system-ui' }}>
      <h1 style={{ color: '#2c3e50', textAlign: 'center' }}>AQI Prediction System</h1>
      
      <form onSubmit={getPrediction} style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
        <select name="City" value={formData.City} onChange={handleChange} style={{ padding: '10px' }}>
          {cities.map(c => <option key={c} value={c}>{c}</option>)}
        </select>

        {['PM2_5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'].map(pollutant => (
          <div key={pollutant}>
            <label>{pollutant.replace('_', '.')}: {formData[pollutant]}</label>
            <input type="range" name={pollutant} min="0" max="300" step={pollutant === 'CO' ? "0.1" : "1"}
                   value={formData[pollutant]} onChange={handleChange} style={{ width: '100%' }} />
          </div>
        ))}

        <button type="submit" style={{ padding: '15px', background: '#27ae60', color: '#fff', border: 'none' }}>
          Predict AQI
        </button>
      </form>

      {error && <p style={{ color: 'red' }}>{error}</p>}
      
      {prediction && (
        <div style={{ marginTop: '20px', padding: '20px', background: '#ecf0f1', textAlign: 'center' }}>
          <h2>AQI: {prediction.predicted_aqi}</h2>
          <h3>Category: {prediction.predicted_category}</h3>
        </div>
      )}
    </div>
  );
}

export default App;