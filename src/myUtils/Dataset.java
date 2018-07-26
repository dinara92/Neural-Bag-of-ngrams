package myUtils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import main.Main;
import myUtils.*;

public class Dataset {
	public List<Text> textList;
	public Dictionary dic = new Dictionary();
	public Set<Integer> labelSet = new HashSet<Integer>();
	//public Map<Integer, Integer> labelMap = new HashMap<Integer, Integer>();

	// MR 10662 sent. dataset
	private List<Text> getMRDataset(String folderPath, int ngram, boolean addData) {
		List<Text> d = new ArrayList<Text>();
		try {
			File[] files = new File(folderPath).listFiles();
		    showFiles(files);
		    File rt_pos = files[0];
		    File rt_neg = files[1];

			BufferedReader reader = new BufferedReader(new FileReader(rt_pos));
			String line = null;
			int index = 0;
			while ((line = reader.readLine()) != null) {
				String key = "train";
				index++;
				int label = 1;
				//labelMap.put(index, label);
				if(index<1066)
					key = "test";
				String content = line.trim();
				//System.out.println(content);
				Text text = new Text(content, key, label, ngram, dic);
				d.add(text);

			}
			reader.close();
			
			BufferedReader reader2 = new BufferedReader(new FileReader(rt_neg));
			line = null;
			//int indexNeg = 0;
			while ((line = reader2.readLine()) != null) {
				String key = "train";
				index++;
				int label = 0;
				//labelMap.put(index, label);

				if(index<5331 + 1066)
					key = "test";
				
				String content = line.trim();
				 //System.out.println(content);
				 Text text = new Text(content, key, label, ngram, dic);
					d.add(text);
			}
			reader2.close();
			System.out.println("indexPos (total number of sent): " + index);
			//System.out.println("indexNeg (total number of sent): " + indexNeg);
			//System.out.println("labelSet size : " + labelMap.size());

		} catch (Exception e) {
			e.printStackTrace();
		}
		return d;
	}
	
	// dataset for training --> for unlabeled dataset (label as -1 in Text text)
	private List<Text> getTrainDataset(String folder, int ngram, int portion) {
		List<Text> d = new ArrayList<Text>();
		try {
			File file = new File(folder);
			BufferedReader reader = new BufferedReader(new FileReader(file));
			String line = null;
			{
				int index = 0;
				while ((line = reader.readLine()) != null) {
					if (index++ % 10000 == 0) {
						System.out.print(".");
					}
					if (index % portion != 0)
						continue;
					String content = line;
					Text text = new Text(content, "dev", -1, ngram, dic);
					d.add(text);
				}
			}
			reader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return d;
	}

	private List<Text>  helperImdbDataset(File file, String key, int label, int ngram) throws IOException{
		BufferedReader reader = new BufferedReader(new FileReader(file));
		String line = null;
		int index = 0;
		List<Text> d = new ArrayList<Text>();

		while ((line = reader.readLine()) != null) {
			index++;
			
			//String content = line.substring(line.split(" ")[0].length()).trim();
			String content = line.trim();
			//System.out.println(content);
			Text text = new Text(content, key, label, ngram, dic);
			d.add(text);
		}
		reader.close();
		return d;	
	}
	
	private List<Text> getImdbDataset(String folderPath, int ngram, boolean addData) throws IOException {
		
			File[] files = new File(folderPath).listFiles();
		    showFiles(files);
			List<Text> d_all = new ArrayList<Text>();

			//imdb
		    File test_neg = files[0];
		    File train_neg = files[1];
		    File train_pos = files[2];
		    File test_pos = files[3];

			//MR
		    /*File test_neg = files[3];
		    File train_neg = files[1];
		    File train_pos = files[0];
		    File test_pos = files[2];*/
		    
		    //CR
		    /*File test_neg = files[0];
		    File train_neg = files[3];
		    File train_pos = files[1];
		    File test_pos = files[2];*/
			
			 //Subj
		    /*File test_neg_subj = files[3];
		    File train_neg_subj = files[0];
		    File train_pos_obj = files[2];
		    File test_pos_obj = files[1];*/
			
		    
		    List<Text> d1 = helperImdbDataset(test_neg, "test", 0, ngram);
		    List<Text> d2 = helperImdbDataset(train_neg, "train", 0, ngram);
		    List<Text> d3 = helperImdbDataset(train_pos, "train", 1, ngram);
		    List<Text> d4 = helperImdbDataset(test_pos, "test", 1, ngram);

		    System.out.println("before total size : " + d_all.size());

		    d_all.addAll(d1);
		    d_all.addAll(d2);
		    d_all.addAll(d3);
		    d_all.addAll(d4);

		    System.out.println("d1 size : " + d1.size());
		    System.out.println("d2 size : " + d2.size());
		    System.out.println("d3 size : " + d3.size());
		    System.out.println("d4 size : " + d4.size());

		    System.out.println("total size : " + d_all.size());
		return d_all;
	}
	
	private List<Text> getImdbDatasetOriginal(String filePath, int ngram, boolean addData) {
		List<Text> d = new ArrayList<Text>();
		try {
			File file = new File(filePath);
			BufferedReader reader = new BufferedReader(new FileReader(file));
			String line = null;
			int readDataCount = 0;
			while ((line = reader.readLine()) != null) {
				String key = "train";
				int index = Integer.parseInt(line.split(" ")[0].substring(2));
				if (25000 <= index && index < 50000)
					key = "test";
				int label = -1;
				if (index < 12500)
					label = 1;
				if (12500 <= index && index < 25000)
					label = 0;
				if (25000 <= index && index < 25000 + 12500)
					label = 1;
				if (25000 + 12500 <= index && index < 25000 + 25000)
					label = 0;
				if (addData) {
					if (label != -1)
						continue;
				} else {
					if (label == -1)
						continue;
				}
				String content = line.substring(line.split(" ")[0].length()).trim();
				// System.out.println(content);
				Text text = new Text(content, key, label, ngram, dic);
				d.add(text);
				if (readDataCount++ % 1000 == 0) {
					System.out.print(".");
				}
			}
			reader.close();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return d;
	}

	public Dataset(String fileName, String type, int ngram, String addDataType) throws IOException {
		System.out.print("reading dataset:" + fileName);
		textList = new ArrayList<Text>();
		if (type.equals("unlabel"))
			textList = getTrainDataset("/home/dinara/java-projects/FusionODPWord2VecNew/clean1BilCorpus/" + fileName, ngram, 1);
		if (type.equals("imdb") || type.equals("MR") || type.equals("CR") || type.equals("Subj"))
			textList = getImdbDataset("./data/" + fileName, ngram, false);
		//if (type.equals("MR_folder"))
		//	textList = getMRDataset("./data/" + fileName, ngram, false);
		
		// some additional unlabeled data
		if (addDataType.contains("news")) {
			String folder = "./data/unlabeled/news.txt";
			List<Text> adtl = getTrainDataset(folder, ngram, Integer.parseInt(addDataType.split("news")[1]));
			for (Text t : adtl) {
				t.type = "dev";
			}
			textList.addAll(adtl);
		}
		if (addDataType.contains("wb")) {
			String folder = "./data/unlabeled/wb.txt";
			List<Text> adtl = getTrainDataset(folder, ngram, Integer.parseInt(addDataType.split("wb")[1]));
			for (Text t : adtl) {
				t.type = "dev";
			}
			textList.addAll(adtl);
		}
		if (addDataType.contains("books")) {
			{
				String folder = "/home/dinara/Downloads/books_in_sentences/books_large_p1.txt"; //not working
				List<Text> adtl = getTrainDataset(folder, ngram, 1);
				for (Text t : adtl) {
					t.type = "dev";
				}
				textList.addAll(adtl);
			}
		}
		if (addDataType.contains("sick")) {
			String folder = "/home/dinara/Downloads/SICK/SICK_train_only.txt"; //not working
			List<Text> adtl = getTrainDataset(folder, ngram, 1);
			for (Text t : adtl) {
				t.type = "dev";
			}
			textList.addAll(adtl);
		}
		if (addDataType.contains("imdbd")) {
			String path = "./data/" + "imdb.txt";
			List<Text> adtl = getImdbDataset(path, ngram, true);
			for (Text t : adtl) {
				t.type = "dev";
			}
			textList.addAll(adtl);
		}

		// set dictionary random word rate
		dic.setRandomFactor(Main.pow);

		System.out.println();
		showDetail();
		System.out.println("reading finished");
	}

	public List<Text> getTextList() {
		return textList;
	}

	public void showDetail() {
		System.out.println("text size = " + textList.size());
		System.out.println("vocab size = " + dic.uniqueWordSize());
		System.out.println("total vocab size = " + dic.totalWordSize);
		double length = 0;
		for (Text text : textList) {
			length += text.getIds(false).size();
		}
		length /= textList.size();
		System.out.println("avg length = " + length);
		double l1 = 0;
		double l2 = 0;
		for (Text text : textList) {
			double t = text.getIds(false).size() - length;
			l1 += Math.abs(t);
			l2 += t * t;
		}
		l1 /= textList.size();
		l2 /= textList.size();
		l2 = Math.sqrt(l2);
		System.out.println("l1 = " + l1);
		System.out.println("l2 = " + l2);
	}
	
	public static void showFiles(File[] files) {
	    for (File file : files) {
	        if (file.isDirectory()) {
	            System.out.println("Directory: " + file.getName());
	            showFiles(file.listFiles()); // Calls same method again.
	        } else {
	            System.out.println("File: " + file.getName());
	        }
	    }
	}
	
}
