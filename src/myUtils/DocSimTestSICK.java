package myUtils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.StringTokenizer;


import model.Model;
import model.WordNSim;

public class DocSimTestSICK {

	public DocSimTestSICK(){


//		this.vec= Word2VecSerializer.ReadWord2Vec();
		
	}

	//public double docSim(String query1, String query2, Model model){
		

		//double [] docVec1 = Model.WordCentroid(query1, model);
		//double [] docVec2 = Model.WordCentroid(query2, model);

		
		//double sim = WordNSim.vectorSim(docVec1, docVec2);

		//return sim;
	//}
	

	public double [] populateSimHuman() throws IOException{
		BufferedReader reader = null;
		double [] simHuman = null;
		int linecount = 0;
		try{
			reader = new BufferedReader(new FileReader("//home//dinara//Downloads//SICK//SICK.txt"));
			String line = reader.readLine();
			simHuman = new double [9841];//9841 doc pairs in SICK dataset
			
			line = reader.readLine(); //miss first line
			while(line!=null){

				System.out.println("this is line " + line);
				StringTokenizer st = new StringTokenizer(line, "\t");
				System.out.println("num of tokens : " + st.countTokens());
				int token_num = st.countTokens();
								
				for(int i = 0; i< token_num; i++){
					String token_score = st.nextToken();
					System.out.println("token : " + token_score);

					if(i == 4){//i==3 if sick-test-only and i==4 if full SICK dataset
						simHuman[linecount] = Double.parseDouble(token_score);
						System.out.println("Rel score human : " + token_score);
					}

				}
				
				line = reader.readLine();
				linecount++;
			}
			

		
		} catch (IOException ioe) {
	        System.out.println(ioe.getMessage());
	    } finally {
	        try { if (reader!=null) reader.close(); } catch (Exception e) {}
	    }
		System.out.println("simHuman size = " + simHuman.length);
		System.out.println(linecount);

		return simHuman;
	    
	}
	
	public double [] populateSimTest(int ngram, int n) throws IOException{
		BufferedReader reader = null;
		double [] simTest = null;
		int linecount = 0;
		try{
			reader = new BufferedReader(new FileReader("//home//dinara//Downloads//SICK//SICK.txt"));
			String line = reader.readLine();
			simTest = new double [9841];//9841 doc pairs in SICK dataset
			
			line = reader.readLine(); //miss first line
			while(line!=null){

				System.out.println("this is line " + line);
				StringTokenizer st = new StringTokenizer(line, "\t");
				System.out.println("num of tokens : " + st.countTokens());
								
				for(int i = 0; i< 4; i++){//i< 3 if sick-test-only and i<4 if full SICK dataset
					//System.out.println("token : " + st.nextToken());
					String token_sent_1 = st.nextToken();
					if(i == 1){
						String token_sent_2 = st.nextToken();
						float sim_current = docSim(token_sent_1, token_sent_2, ngram, n);
						simTest[linecount] = sim_current;
						System.out.println("Rel score : " + sim_current);

					}
					
				}
				
				line = reader.readLine();
				linecount++;
			}
			

		
		} catch (IOException ioe) {
	        System.out.println(ioe.getMessage());
	    } finally {
	        try { if (reader!=null) reader.close(); } catch (Exception e) {}
	    }
		System.out.println("simHuman size = " + simTest.length);
		System.out.println(linecount);

		return simTest;
	    
	}
	
	
	public static float docSim(String query1, String query2, int ngram, int n){
		float [] docVec1 = Model.WordCentroid(query1,  ngram, n);
		float [] docVec2 = Model.WordCentroid(query2,  ngram, n);

		float sim = WordNSim.vectorSim(docVec1, docVec2);
		return sim;
	}

}

