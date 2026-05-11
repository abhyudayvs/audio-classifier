import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import { Mic, Square, UploadCloud, Activity, Clock, AlertCircle } from 'lucide-react';
import RecordRTC from 'recordrtc';

export default function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [recording, setRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  const mediaRecorderRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    const saved = localStorage.getItem('aiAudioHistory');
    if (saved) setHistory(JSON.parse(saved));
  }, []);

  const saveToHistory = (res) => {
    const newHistory = [{ id: Date.now(), ...res }, ...history].slice(0, 50); // Keep last 50
    setHistory(newHistory);
    localStorage.setItem('aiAudioHistory', JSON.stringify(newHistory));
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      // Forces perfect 16kHz WAV format (Required for Edge Impulse)
      mediaRecorderRef.current = new RecordRTC(stream, {
        type: 'audio',
        mimeType: 'audio/wav',
        recorderType: RecordRTC.StereoAudioRecorder,
        desiredSampRate: 16000,
        numberOfAudioChannels: 1 
      });

      mediaRecorderRef.current.startRecording();
      setRecording(true);
      setError('');
    } catch (err) {
      setError('Microphone access denied. Please allow microphone permissions in your browser.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stopRecording(() => {
        const audioBlob = mediaRecorderRef.current.getBlob();
        const url = URL.createObjectURL(audioBlob);
        setAudioUrl(url);

        const file = new File([audioBlob], "microphone_recording.wav", { type: 'audio/wav' });
        handleAnalyze(file);

        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
        }
      });
    }
    setRecording(false);
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setAudioUrl(URL.createObjectURL(file));
      handleAnalyze(file);
    }
  };

  const handleAnalyze = async (fileToUpload) => {
    setLoading(true);
    setError('');
    setResult(null);
    
    const formData = new FormData();
    formData.append('file', fileToUpload);

    try {
      const response = await axios.post('http://localhost:8000/predict', formData);
      setResult(response.data);
      saveToHistory({ 
        filename: fileToUpload.name, 
        prediction: response.data.prediction, 
        date: new Date().toLocaleString() 
      });
      setActiveTab('results');
    } catch (err) {
      setError(err.response?.data?.detail || 'Cannot connect to AI Backend. Is your Python server running?');
    } finally {
      setLoading(false);
    }
  };

  const chartData = result ? Object.keys(result.confidence).map(key => ({
    name: key,
    value: parseFloat((result.confidence[key] * 100).toFixed(1))
  })) : [];

  return (
    <div className="flex h-screen bg-gray-50 text-gray-800 font-sans">
      {/* Sidebar */}
      <div className="w-64 bg-white border-r shadow-sm flex flex-col p-6">
        <h1 className="text-2xl font-black text-indigo-700 mb-8 flex items-center gap-3">
          <Activity size={28} /> AI Audio
        </h1>
        <nav className="flex flex-col gap-2">
          {['upload', 'record', 'results', 'history'].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`p-3 text-left rounded-lg capitalize font-bold transition-all ${
                activeTab === tab ? 'bg-indigo-600 text-white shadow-md' : 'hover:bg-indigo-50 text-gray-600'
              }`}
            >
              {tab}
            </button>
          ))}
        </nav>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 p-10 overflow-y-auto">
        {error && (
          <div className="flex items-center gap-2 p-4 mb-6 bg-red-100 text-red-700 font-medium rounded-lg border border-red-200">
            <AlertCircle /> {error}
          </div>
        )}
        
        {loading && (
          <div className="flex items-center gap-3 text-indigo-600 mb-6 font-bold text-xl animate-pulse">
            <Activity className="animate-spin" size={28} /> AI is analyzing audio stream...
          </div>
        )}

        {/* Upload Tab */}
        {activeTab === 'upload' && (
          <div className="max-w-3xl bg-white p-10 rounded-2xl shadow-sm border border-gray-100">
            <h2 className="text-3xl font-black mb-4 text-gray-800">Upload File</h2>
            <p className="text-gray-500 mb-8 text-lg">Upload a .wav audio file for the AI to classify.</p>
            <label className="flex flex-col items-center justify-center h-64 border-4 border-dashed border-indigo-200 bg-indigo-50/50 rounded-xl cursor-pointer hover:bg-indigo-50 hover:border-indigo-400 transition-all">
              <UploadCloud size={64} className="text-indigo-500 mb-4" />
              <span className="font-bold text-indigo-700 text-xl">Click to Browse</span>
              <input type="file" accept=".wav" className="hidden" onChange={handleFileUpload} />
            </label>
          </div>
        )}

        {/* Record Tab */}
        {activeTab === 'record' && (
          <div className="max-w-3xl bg-white p-10 rounded-2xl shadow-sm border border-gray-100">
            <h2 className="text-3xl font-black mb-4 text-gray-800">Live Microphone</h2>
            <p className="text-gray-500 mb-8 text-lg">Record a sound directly through your browser.</p>
            <div className="flex flex-col items-center justify-center h-64 bg-gray-50 rounded-xl border-2 border-gray-100">
              {recording ? (
                <div className="flex flex-col items-center">
                  <span className="flex h-6 w-6 relative mb-6">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-6 w-6 bg-red-600"></span>
                  </span>
                  <button onClick={stopRecording} className="flex items-center gap-2 px-8 py-4 bg-red-500 text-white font-black text-lg rounded-full hover:bg-red-600 shadow-lg hover:shadow-red-500/30 transition-all hover:-translate-y-1">
                    <Square size={20} /> Stop & Analyze
                  </button>
                </div>
              ) : (
                <button onClick={startRecording} className="flex items-center gap-2 px-8 py-4 bg-indigo-600 text-white font-black text-lg rounded-full hover:bg-indigo-700 shadow-lg hover:shadow-indigo-500/30 transition-all hover:-translate-y-1">
                  <Mic size={20} /> Start Recording
                </button>
              )}
            </div>
          </div>
        )}

        {/* Results Tab */}
        {activeTab === 'results' && (
          <div className="max-w-5xl space-y-8">
            <h2 className="text-3xl font-black text-gray-800">Analysis Complete</h2>
            {audioUrl && <audio src={audioUrl} controls className="w-full rounded-lg shadow-sm" />}
            
            {result ? (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-gradient-to-br from-indigo-600 to-violet-700 text-white p-8 rounded-2xl shadow-lg flex flex-col justify-center items-center text-center">
                    <p className="text-indigo-200 font-bold mb-2 uppercase tracking-wider text-sm">Overall Prediction</p>
                    <h3 className="text-5xl font-black capitalize leading-tight">{result.prediction.replace('_', ' ')}</h3>
                  </div>
                  
                  <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 h-72">
                    <h4 className="font-bold mb-6 text-gray-800 tracking-wide text-sm uppercase">Peak Confidence Scores</h4>
                    <ResponsiveContainer width="100%" height="85%">
                      <BarChart data={chartData} layout="vertical" margin={{ top: 0, right: 30, left: 40, bottom: 0 }}>
                        <XAxis type="number" hide domain={[0, 100]} />
                        <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} tick={{fontWeight: 'bold', fill: '#4b5563'}} />
                        <Tooltip cursor={{ fill: '#f3f4f6' }} contentStyle={{borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)'}} />
                        <Bar dataKey="value" fill="#4f46e5" radius={[0, 6, 6, 0]} barSize={24} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {result.counts && (
                  <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100">
                    <h4 className="font-bold mb-6 text-gray-800 tracking-wide text-sm uppercase">Event Counter</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      {Object.entries(result.counts).map(([label, count]) => (
                        <div key={label} className={`p-6 rounded-xl text-center transition-colors ${count > 0 ? 'bg-emerald-50 border-emerald-200 border-2' : 'bg-gray-50 border border-gray-100'}`}>
                          <div className={`text-4xl font-black ${count > 0 ? 'text-emerald-600' : 'text-gray-300'}`}>{count}</div>
                          <div className={`text-sm mt-2 capitalize font-bold ${count > 0 ? 'text-emerald-800' : 'text-gray-400'}`}>{label.replace('_', ' ')}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
               <div className="p-10 text-center border-2 border-dashed border-gray-200 rounded-2xl text-gray-400 font-medium text-lg">
                 No results generated yet. Please upload or record an audio sample.
               </div>
            )}
          </div>
        )}

        {/* History Tab */}
        {activeTab === 'history' && (
          <div className="max-w-5xl">
            <h2 className="text-3xl font-black mb-8 flex items-center gap-3 text-gray-800"><Clock size={28} /> Session History</h2>
            <div className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
              <table className="w-full text-left">
                <thead className="bg-gray-50/80 border-b border-gray-100">
                  <tr>
                    <th className="p-5 font-bold text-gray-500 uppercase tracking-wider text-xs">Timestamp</th>
                    <th className="p-5 font-bold text-gray-500 uppercase tracking-wider text-xs">Source File</th>
                    <th className="p-5 font-bold text-gray-500 uppercase tracking-wider text-xs">AI Prediction</th>
                  </tr>
                </thead>
                <tbody>
                  {history.length > 0 ? history.map(item => (
                    <tr key={item.id} className="border-b border-gray-50 last:border-0 hover:bg-gray-50/50 transition-colors">
                      <td className="p-5 text-gray-500 text-sm">{item.date}</td>
                      <td className="p-5 font-medium text-gray-700">{item.filename}</td>
                      <td className="p-5 text-indigo-600 font-black capitalize">{item.prediction.replace('_', ' ')}</td>
                    </tr>
                  )) : (
                    <tr><td colSpan="3" className="p-10 text-center text-gray-400 font-medium">No previous predictions found.</td></tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}