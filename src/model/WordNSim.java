package model;

public class WordNSim {
	private String word;
	private double sim;
	
	public WordNSim(String word, double sim){
		this.word = word;
		this.sim = sim;
	}
	public String getWord(){
		return this.word;
	}
	public double getSim(){
		return this.sim;
	}
	
	public static float vectorSim(float[] v1, float[] v2){
		float sim = 0;
		for(int i=0;i<v1.length;i++)
			sim +=v1[i]*v2[i];
		return sim;
	}
}