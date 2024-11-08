import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import { FaRobot, FaUser } from "react-icons/fa";
import "./App.css";

const ImageGenerator = () => {
  const [messages, setMessages] = useState([
    { text: "Welcome! Describe the image you want, and I will generate it for you.", type: "bot" },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const endOfMessagesRef = useRef(null);

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async (e) => {
    e.preventDefault();
    if (input.trim()) {
      setMessages([...messages, { text: input, type: "user" }]);
      setInput("");
      setLoading(true);

      try {
        const response = await axios.post("https://active-mallard-uniquely.ngrok-free.app/generate_image_and_description", { prompt: input });
        const { image, description } = response.data;
        setMessages(prevMessages => [
          ...prevMessages,
          { text: "Here is your generated image:", type: "bot" },
          { text: <img src={`data:image/png;base64,${image}`} alt="Generated" className="generated-image" />, type: "bot" },
          { text: description, type: "bot" },
        ]);
      } catch (error) {
        console.error("Error generating image:", error);
        setMessages(prevMessages => [
          ...prevMessages,
          { text: "Sorry, something went wrong. Please try again later.", type: "bot" },
        ]);
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white">
      <header className="bg-gray-800 p-4 shadow-md flex justify-center items-center">
        <h1 className="text-2xl font-bold">AI Image Generator</h1>
      </header>
      <main className="flex-1 p-4 overflow-hidden hide-scrollbar">
        <div className="flex flex-col space-y-4 h-full overflow-auto">
          {messages.map((msg, index) => (
            <div key={index} className={`flex ${msg.type === "user" ? "justify-end" : "justify-start"} items-start`}>
              <div className="flex items-center">
                {msg.type === "bot" ? <FaRobot className="text-gray-300" size={24} /> : <FaUser className="text-gray-300" size={24} />}
              </div>
              <div className={`p-3 rounded-lg w-auto max-w-xs sm:max-w-md md:max-w-lg lg:max-w-xl xl:max-w-2xl ml-2 text-left ${msg.type === "user" ? "bg-blue-500 text-white" : "bg-gray-700 text-white"} break-words`}>
                {msg.text}
              </div>
            </div>
          ))}
          <div ref={endOfMessagesRef} />
        </div>
      </main>
      <footer className="p-4 bg-gray-800">
        <div className="flex justify-center">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") handleSend(e); }}
            disabled={loading}
            className="w-2/3 p-2 rounded-l-lg bg-gray-100 text-gray-800"
            placeholder="Describe the image you want to generate..."
          />
          <button
            onClick={handleSend}
            disabled={loading}
            className="bg-blue-500 text-white px-5 rounded-r-lg hover:bg-blue-700"
          >
            {loading ? "Generating..." : "Send"}
          </button>
        </div>
      </footer>
    </div>
  );
};

export default ImageGenerator;
