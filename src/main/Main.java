package main;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import org.apache.commons.math3.stat.correlation.SpearmansCorrelation;

import model.Model;
import model.Task;
import myUtils.*;
import myUtils.Dictionary;

public class Main {
	// all the parameters are here
	public static int ngram = 2;//original ngram = 1
	public static float lr = 0.025f; // learning rate //0.1 89.93420.0
	public static float rr = 0.000f; // regulize rate //not used
	public static int negSize = 5; // negative sampling size
	public static int iter = 10; // iteration
	public static int batchSize = 100; //original batchSize = 100
	public static int ws = 5; // CGNR window size
	public static int n = 500; // word embeddings dimension
	public static String info = "d"; // w, d or l // w stands for CGNR. d stands
										// for TGNR. l stands for LGNR.
	public static boolean useUniqueWordList = false;
	public static String negType = "i"; // i or o. negative sampling type
	public static boolean preLogistic = false;
	public static boolean saveWordVectors = true;
	public static double subSampleRate = 0;
	public static double dropRate = 0;
	public static String addDataType = ""; // news4
	public static double subRate = 1;
	public static double pow = 1;
	public static int testId = new Random().nextInt(10000);

	public static class DatasetTask {
		public String folderName;
		public String type;

		public DatasetTask(String folderName, String type) {
			this.folderName = folderName;
			this.type = type;
		}
	}

	public static void train(List<DatasetTask> taskList) {

		for (int index_task = 0; index_task < taskList.size(); index_task++) {
			DatasetTask task = taskList.get(index_task);

			// load dataset
			Dataset dataset = null;
			try {
				dataset = new Dataset(task.folderName, task.type, ngram, addDataType);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			// initialize model
			Model model = new Model(task.folderName, dataset, lr, rr, negSize, iter, batchSize, ws, n, info);

			System.out.println(index_task + "||" + index_task + "||" + index_task + "||" + index_task + "||");
			// train model
			model.train();
			
			{
				long startTime = System.currentTimeMillis();

				// save model
				String baseTmpSaveFolder = "./results/" + task.folderName + ngram + "/" + info + "_" + addDataType
						+ "_" + n + "_r" + testId + "/";
				String tmpSaveFolder = baseTmpSaveFolder + "fold" + index_task + "/";
				if (saveWordVectors)
					model.saveWordVectors(tmpSaveFolder);

				System.out.println("||" + "|" + "time:" + (System.currentTimeMillis() - startTime));
			}
		}
		System.out.println();
		System.out.println(info);
		System.out.println();
	}


	public static void loadModel() throws IOException{
		
	
		/*for (int index_task = 0; index_task < taskList.size(); index_task++) {
			DatasetTask task = taskList.get(index_task);

			// load dataset
			Dataset dataset = null;
			try {
				dataset = new Dataset(task.folderName, task.type, ngram, addDataType);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
	
			// load model
			Model model = new Model(task.folderName, dataset, lr, rr, negSize, iter, batchSize, ws, n, info);
			//model.resume(model);
			//model.nearestWords(model, "good_", 20);
			//model.docSimMain(model);
			//model.printWordVectors(model);
			
		}*/
		Model model = new Model(n);///home/dinara/java-projects/Neural-BoN-master/results/news.en-0000-of-00100.txt1/d__500_r8946/fold0
		//HashMap<String, float[]> wordVM = model.loadWordVectors("./results/MR2/d__500_r649/fold0/", n);
		model.docSimMain(ngram, n);
		//model.nearestWords("not_good_", 10);
	}
	
	
	public static void main(String args[]) throws IOException {

		{
			List<DatasetTask> taskList = new ArrayList<DatasetTask>();
			/**
			 * choose one of following line to add a task.
			 * 
			 * "unlabel" indicate unlabeled corpus and suitable only for CGNR
			 * and TGNR.
			 * 
			 * "imdb" indicate imdb format. You can implement an getXXXDataset function
			 * in src/myUtils/Dataset.java our change the getIMDBDataset
			 * function in src/myUtils/Dataset.java for the specific format.
			 */
			 taskList.add(new DatasetTask("news.en-0000-of-00100.txt", "unlabel"));

			//taskList.add(new DatasetTask("imdb.txt", "imdb"));
			//taskList.add(new DatasetTask("imdb_train_test", "imdb"));
			//taskList.add(new DatasetTask("MR", "MR"));
			//taskList.add(new DatasetTask("CR", "CR"));
			//taskList.add(new DatasetTask("Subj", "Subj"));

			//taskList.add(new DatasetTask("MR", "MR_folder"));
			//train(taskList);
			loadModel();
					}
	}
}
