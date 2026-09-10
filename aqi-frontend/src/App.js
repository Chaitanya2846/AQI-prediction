import React, { useState } from 'react';
import './App.css';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');
  
  const cities = ['agartala', 'ahmedabad', 'aizawl', 'bengaluru', 'bhopal', 'bhubaneswar', 
    'chandigarh', 'chennai', 'dehradun', 'delhi', 'gangtok', 'gurugram', 
    'guwahati', 'hyderabad', 'imphal', 'itanagar', 'jaipur', 'kohima', 
    'kolkata', 'lucknow', 'mumbai', 'panaji', 'patna', 'raipur', 'ranchi', 
    'shillong', 'shimla', 'thiruvananthapuram', 'visakhapatnam'];
  const [formData, setFormData] = useState({
    City: cities[0], 
    PM2_5: 45.0, PM10: 90.0, NO2: 15.0, SO2: 20.0, CO: 1.0, O3: 50.0,
    Temperature: 28.0, Humidity: 60.0, Wind_Speed: 10.0,
    Crop_Burning: 0, Festival: 0
  });  
  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({ 
        ...formData, 
        [name]: type === 'checkbox' ? (checked ? 1 : 0) : (name === 'City' ? value : parseFloat(value)) 
    });
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
    <div style={{ maxWidth: '700px', margin: '2rem auto', fontFamily: 'system-ui, sans-serif' }}>
      <h1 style={{ color: '#2c3e50', textAlign: 'center' }}>Advanced AQI Prediction System</h1>
      
      <form onSubmit={getPrediction} style={{ display: 'flex', flexDirection: 'column', gap: '15px', padding: '20px', border: '1px solid #ddd', borderRadius: '8px' }}>
        
        <div>
            <label style={{fontWeight: 'bold'}}>Select City: </label>
            <select name="City" value={formData.City} onChange={handleChange} style={{ padding: '8px', width: '100%', marginTop: '5px' }}>
            {cities.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
        </div>

        <h3 style={{ margin: '10px 0 0 0', color: '#7f8c8d', borderBottom: '1px solid #eee', paddingBottom: '5px' }}>Pollutant Levels</h3>
        {['PM2_5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'].map(pollutant => (
          <div key={pollutant}>
            <label style={{fontSize: '14px'}}>{pollutant.replace('_', '.')}: <b>{formData[pollutant]}</b></label>
            <input type="range" name={pollutant} min="0" max="300" step={pollutant === 'CO' ? "0.1" : "1"}
                value={formData[pollutant]} onChange={handleChange} style={{ width: '100%', cursor: 'pointer' }} />
          </div>
        ))}

        <h3 style={{ margin: '10px 0 0 0', color: '#7f8c8d', borderBottom: '1px solid #eee', paddingBottom: '5px' }}>Meteorological Data</h3>
        <div>
            <label style={{fontSize: '14px'}}>Temperature (°C): <b>{formData.Temperature}</b></label>
            <input type="range" name="Temperature" min="0" max="50" step="0.5" value={formData.Temperature} onChange={handleChange} style={{ width: '100%' }} />
        </div>
        <div>
            <label style={{fontSize: '14px'}}>Humidity (%): <b>{formData.Humidity}</b></label>
            <input type="range" name="Humidity" min="0" max="100" step="1" value={formData.Humidity} onChange={handleChange} style={{ width: '100%' }} />
        </div>
        <div>
            <label style={{fontSize: '14px'}}>Wind Speed (km/h): <b>{formData.Wind_Speed}</b></label>
            <input type="range" name="Wind_Speed" min="0" max="50" step="1" value={formData.Wind_Speed} onChange={handleChange} style={{ width: '100%' }} />
        </div>

        <h3 style={{ margin: '10px 0 0 0', color: '#7f8c8d', borderBottom: '1px solid #eee', paddingBottom: '5px' }}>Environmental Context</h3>
        <div style={{ display: 'flex', gap: '20px' }}>
            <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <input type="checkbox" name="Crop_Burning" checked={formData.Crop_Burning === 1} onChange={handleChange} />
                Active Crop Burning
            </label>
            <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <input type="checkbox" name="Festival" checked={formData.Festival === 1} onChange={handleChange} />
                Major Festival Period
            </label>
        </div>

        <button type="submit" style={{ padding: '15px', background: '#27ae60', color: '#fff', border: 'none', borderRadius: '5px', fontSize: '16px', cursor: 'pointer', marginTop: '15px' }}>
          Predict AQI
        </button>
      </form>

      {prediction && (
        <div style={{ marginTop: '20px', padding: '20px', background: '#ecf0f1', textAlign: 'center', borderRadius: '8px', border: '2px solid #27ae60' }}>
          <h2 style={{ margin: '0 0 10px 0' }}>Predicted AQI: {prediction.predicted_aqi}</h2>
          <h3 style={{ margin: '0', color: '#34495e' }}>Category: {prediction.predicted_category}</h3>
        </div>
      )}
    </div>
  );
}

export default App;