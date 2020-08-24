package eu.everis.hercules.sparql.mapping;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.hp.hpl.jena.query.*;
import com.hp.hpl.jena.sparql.engine.http.QueryEngineHTTP;
import com.jayway.jsonpath.JsonPath;
import com.squareup.okhttp.OkHttpClient;
import com.squareup.okhttp.Request;
import com.squareup.okhttp.RequestBody;
import com.squareup.okhttp.Response;
import org.apache.commons.lang3.StringUtils;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import java.io.*;
import java.net.URLEncoder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.lang.Thread.sleep;

/**
 * SPARQL Mapper
 *
 */
public class SPARQLMapper
{
	private static int rowNum = 1;
	private static int maxHeight = 1;
	private static final String UNDERSCORE = "_";

	private static int keywordCount = 0;
	private static final String CHEMBL_STRING = "chembl";
	private static final String AGROVOC_STRING = "agrovoc";
	private static final String MESH_STRING = "mesh";
	private static final String DBPEDIA_STRING = "dbPedia";
	private static final String EUROSCIVOC_STRING = "euroSciVoc";
	private static int countOfMappedKeywords = 0;

	public static void main(String[] args) throws Exception
	{
		//processCorpusFile("C:\\Documents\\Hercules\\Mapping\\Exp3-Input\\BIO\\lemasAbstract\\AUTHOR_WDcorpus_tfidf.txt", "","","C:\\Documents\\Hercules\\Mapping\\Exp3-Output\\withEuroSci\\BIO\\lemasAbstract\\AUTHOR_WDcorpus_tfidf.xlsx");
		System.out.println("Processing input file: "+args[0] );
		System.out.println("args[1]: "+args[1] );
		System.out.println("args[2]: "+args[2] );
		String inputFile = args[0];
		String outputLocation = args[1];
		String documentType = args[2];
		File file = new File(inputFile);
		String inputFileName = file.getName();
		// Only process WDcorpus files
		if (inputFileName.indexOf(".") > 0) {
			String fileName = inputFileName.substring(0, inputFileName.lastIndexOf("."));
			String parentDir = file.getParentFile().getName();
			String outputDir =  outputLocation+"\\"+parentDir;
			String outputFile = outputDir +"\\"+fileName+"-mappings.xlsx";
			File directory = new File(outputDir);
			if(!directory.exists()){
				directory.mkdir();
			}
			processCorpusFile(args[0], "","",outputFile, documentType);
		}
	}

	private static void processCorpusFile(String corpusFileName, String keywordFileName, String keywordReplacedCorpusFile, String mappingExcelFile, String documentType) throws Exception{
		//extract keywords into a map
		if(!keywordFileName.isEmpty()) {
			HashMap<String, String> keywordsMap = extractKeywords(keywordFileName);
			processCorpusAndProduceKeywordReplacedFile(corpusFileName, keywordsMap, keywordReplacedCorpusFile);
		}

		// for AGR and non-author documents, the documentId is at index 1, else it is index 0
		int documentIdx = ("AGR".equals(documentType) && !(corpusFileName.contains("AUTHOR") || corpusFileName.contains("author"))) ? 1 : 0;
		LinkedHashMap<String, Map<String, Double>> documentMap = processKeywordReplacedCorpusFile(!keywordReplacedCorpusFile.isEmpty() ? keywordReplacedCorpusFile : corpusFileName, documentIdx);
		performMappingForDocuments(documentMap, mappingExcelFile);
		System.out.println("\nProcessing completed for input file:"+corpusFileName);
		System.out.println("\nOutput file location:"+mappingExcelFile);
		printStatistics();
	}

	private static LinkedHashMap<String, Map<String, Double>> processKeywordReplacedCorpusFile(String keywordReplacedCorpusFile, int documentIdx) {
		String words[];
		String line;
		String documentId = "";
		HashMap<String, Double> keywordScoreMap = new HashMap<String, Double>();
		LinkedHashMap<String, Map<String, Double>> documentMap = new LinkedHashMap<String, Map<String, Double>>();

		try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(keywordReplacedCorpusFile), "UTF-8")))
		{
			while(((line = br.readLine()) != null)){
				keywordScoreMap.clear();
				words = line.split("\\s");
				if(words.length > documentIdx){
					documentId = words[documentIdx];
				}
				System.out.println("Processing document entry "+documentId);
				for (int i = documentIdx+1; i < words.length; i++) {
					if(words[i].contains(":")){
						int index = words[i].lastIndexOf(":");
						String keyword = words[i].substring(0, index);
						String scoreStr = words[i].substring(index+1);
						//Ignore keywords which are http links
						if(!keyword.startsWith("http")){
							keywordScoreMap.put(keyword, Double.parseDouble(scoreStr));
						}
					}
				}

				TreeMap<String, Double> sorted_map = sortMapByScore(keywordScoreMap);

				documentMap.put(documentId, sorted_map);

			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return documentMap;
	}

	private static void printStatistics() {
		System.out.println("Keyword Count: "+keywordCount);
		System.out.println("Total Mapped Keywords: "+countOfMappedKeywords);
		System.out.println("Percentage of Mapped Keywords: " + (Math.round(countOfMappedKeywords*100/keywordCount)));

	}

	private static void processCorpusAndProduceKeywordReplacedFile(String corpusFileName, Map keywordsMap, String keywordReplacedCorpusFile) throws Exception{
		String words[];
		String line;
		FileWriter fw = new FileWriter(keywordReplacedCorpusFile);
		try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(corpusFileName), "UTF-8")))
		{
			while(((line = br.readLine()) != null)){
				words = line.split("\\s");
				StringBuilder sb = new StringBuilder("");
				sb.append(words[0]);
				sb.append(' ');
				sb.append(words[1]);
				sb.append(' ');
				for (int i = 2; i < words.length; i++) {
					String[] split = words[i].split(":");
					String keywordId = split[0];
					String keyword = (String)keywordsMap.get(keywordId);
					sb.append(keyword);
					sb.append(":");
					sb.append(split[1]);
					sb.append(' ');

				}
				sb.append(System.lineSeparator());
				fw.write(sb.toString());
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		fw.close();
	}
	public static class Mapped {
		private String score;

		public String getKeyword() {
			return keyword;
		}

		public void setKeyword(String keyword) {
			this.keyword = keyword;
		}

		private String keyword;

		public Mapped(String keyword){
			this.keyword = keyword;
		}
		public String getScore() {
			return score;
		}

		public void setScore(String score) {
			this.score = score;
		}

		public List<String> getAgrovocURI() {
			return agrovocURI;
		}

		public void setAgrovocURI(List<String> agrovocURI) {
			this.agrovocURI = agrovocURI;
		}

		public List<String> getChemblURI() {
			return chemblURI;
		}

		public void setChemblURI(List<String> chemblURI) {
			this.chemblURI = chemblURI;
		}

		public List<String> getMeshURI() {
			return meshURI;
		}

		public void setMeshURI(List<String> meshURI) {
			this.meshURI = meshURI;
		}

		public List<String> getDbPediaURI() {
			return dbPediaURI;
		}

		public void setDbPediaURI(List<String> dbPediaURI) {
			this.dbPediaURI = dbPediaURI;
		}

		public List<String> getEuroSciVocURI() {
			return euroSciVocURI;
		}

		public void setEuroSciVocURI(List<String> euroSciVocURI) {
			this.euroSciVocURI = euroSciVocURI;
		}

		private List<String> agrovocURI;
		private List<String> chemblURI;
		private List<String> meshURI;
		private List<String> dbPediaURI;
		private List<String> euroSciVocURI;



	}

	private static void performMappingForDocuments(LinkedHashMap<String, Map<String, Double>> documentMap, String mappingExcelFile) throws Exception{
		String[] columns = new String[]{"DocumentId", "Keyword", "Score","Agrovoc URIs", "Mesh URIs", "Chembl URIs",  "DBPedia URIs", "EuroSciVoc URIs"};

		Workbook workbook = createExcelWorkbook(columns);
		Iterator itDocuments = documentMap.entrySet().iterator();
		ExecutorService executor = Executors.newFixedThreadPool(documentMap.size() < 300 ? documentMap.size() : 300);
		List<Future<?>> futures = new ArrayList<Future<?>>();
		LinkedHashMap<String, List<Mapped>> mappedMap = new LinkedHashMap<String, List<Mapped>>();

		while (itDocuments.hasNext()) {


			Map.Entry documentEntry = (Map.Entry)itDocuments.next();
			String documentId = (String)documentEntry.getKey();
			System.out.println("Processing method performMappingForDocuments for documentId: "+documentId);
			TreeMap keywordsMap = (TreeMap)documentEntry.getValue();
			List<Mapped> mappedList = new ArrayList<Mapped>();
			mappedMap.put(documentId, mappedList);
			Runnable documentKeywordMapper = new DocumentKeywordMapperRunnable(documentId, keywordsMap, mappedMap);
			Future<?> future = executor.submit(documentKeywordMapper);
			futures.add(future);

		}
		// Wait until all threads are finished
		for(Future<?> future : futures)
			future.get();

		executor.shutdownNow();
		Iterator itKeywordsMapping = mappedMap.entrySet().iterator();
		while (itKeywordsMapping.hasNext()) {
			Map.Entry keywordMap = (Map.Entry)itKeywordsMapping.next();
			String documentId = (String)keywordMap.getKey();
			List<Mapped> listMappedObjects = (List<Mapped>)keywordMap.getValue();
			for(Mapped mapped  : listMappedObjects){
				List<String> chemblUris = mapped.getChemblURI();
				List<String> agrovocUris = mapped.getAgrovocURI();
				List<String> meshUris = mapped.getMeshURI();
				List<String> dbPediaUris = mapped.getDbPediaURI();
				List<String> euroSciVocUris = mapped.getEuroSciVocURI();
				if(chemblUris.size() > 0 || agrovocUris.size() > 0 || meshUris.size() > 0 || dbPediaUris.size()>0 || euroSciVocUris.size()>0){
					countOfMappedKeywords++;
				}
				writeToExcel(workbook, columns, documentId, mapped.getKeyword(), mapped.getScore(), agrovocUris, meshUris, chemblUris, dbPediaUris, euroSciVocUris);
			}
		}
		closeWorkbook(workbook, mappingExcelFile);

	}

	public static class DocumentKeywordMapperRunnable implements Runnable {
		private String documentId;
		private TreeMap keywordsMap;
		LinkedHashMap<String, List<Mapped>> mappedMap;
		LinkedHashMap<String, HashMap<String, List<String>>> keywordsMappings;

		DocumentKeywordMapperRunnable(String documentId, TreeMap keywordsMap, LinkedHashMap<String, List<Mapped>> mappedMap) {
			this.documentId = documentId;
			this.keywordsMap = keywordsMap;
			this.mappedMap = mappedMap;
		}

		@Override
		public void run() {

			try {
				int i = 0;
				List<Mapped> mappedList = this.mappedMap.get(documentId);
				Iterator itKeywords = this.keywordsMap.entrySet().iterator();
				int minValue = this.keywordsMap.size() < 20 ? this.keywordsMap.size() : 20;
				LinkedHashMap<String, HashMap<String, List<String>>> keywordsMappings = new LinkedHashMap<String, HashMap<String, List<String>>>();
				while(i < minValue ){
					i++;
					keywordCount++;
					Map.Entry keywordEntry = (Map.Entry)itKeywords.next();
					String keyword = (String)keywordEntry.getKey();
					Double score = (Double)keywordEntry.getValue();
					String keywordScoreString = keyword + ":" + score.toString();
					keywordsMappings.put(keywordScoreString, new HashMap<String, List<String>>() );
					//List<String> chemblUris = findInChembl(keyword);
					mapKeywords(keywordScoreString, keywordsMappings);
				}
				Iterator itKeywordsMapping = keywordsMappings.entrySet().iterator();
				while (itKeywordsMapping.hasNext()) {
					Map.Entry keywordMap = (Map.Entry)itKeywordsMapping.next();
					String keywordScore = (String)keywordMap.getKey();
					String[] keys = keywordScore.split(":");
					HashMap<String, List<String>> keywordMaps = (HashMap<String, List<String>>)keywordMap.getValue();
					List<String> chemblUris = keywordMaps.get(CHEMBL_STRING);
					List<String> agrovocUris = keywordMaps.get(AGROVOC_STRING);
					List<String> meshUris = keywordMaps.get(MESH_STRING);
					List<String> dbPediaUris = keywordMaps.get(DBPEDIA_STRING);
					List<String> euroSciVocUris = keywordMaps.get(EUROSCIVOC_STRING);
					Mapped mapped = new Mapped(keys[0]);
					mapped.setScore(keys[1]);
					mapped.setAgrovocURI(agrovocUris);
					mapped.setMeshURI(meshUris);
					mapped.setChemblURI(chemblUris);
					mapped.setDbPediaURI(dbPediaUris);
					mapped.setEuroSciVocURI(euroSciVocUris);
					mappedList.add(mapped);
				}
				this.mappedMap.put(documentId, mappedList);
			} catch (Exception e){
				System.err.println("Error in processing  ");
			}

		}
	}

	public static void mapKeywords(String keywordScore, LinkedHashMap<String, HashMap<String, List<String>>> keywordsMappings) {

		try {
			String[] keys = keywordScore.split(":");
			String keyword = keys[0].trim();
			System.out.println("Processing keyword " + keyword);
			List<String> chemblUris = new ArrayList<String>();
			List<String> euroSciVocUris = findInEuroSciVoc(keyword);
			List<String> agrovocUris = findInAgrovoc(keyword);
			List<String> meshUris = findInMesh(keyword);
			List<String> dbPediaUris = findInDBPedia(keyword);
			HashMap<String, List<String>> mappings = keywordsMappings.get(keywordScore);
			mappings.put(CHEMBL_STRING, chemblUris);
			mappings.put(AGROVOC_STRING, agrovocUris);
			mappings.put(MESH_STRING, meshUris);
			mappings.put(DBPEDIA_STRING, dbPediaUris);
			mappings.put(EUROSCIVOC_STRING, euroSciVocUris);
			keywordsMappings.put(keywordScore, mappings);

		} catch (Exception e) {
			System.err.println("Error in processing keyword " + keywordScore);
		}
	}
	private static TreeMap<String, Double> sortMapByScore(Map<String, Double> keywordScoreMap) {
		ScoreComparator bvc = new ScoreComparator(keywordScoreMap);
		TreeMap<String, Double> sorted_map = new TreeMap<String, Double>(bvc);
		sorted_map.putAll(keywordScoreMap);
		return sorted_map;
	}

	private static HashMap<String, String> extractKeywords(String fileName) {
		String line;
		HashMap<String, String> keywordMap = new HashMap<String, String>();
		try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8")))
		{
			while(((line = br.readLine()) != null)){
				String[] words = line.split(":");
				keywordMap.put(words[0], words[1]);

			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return keywordMap;
	}

	static class ScoreComparator implements Comparator<String> {
		Map<String, Double> base;

		public ScoreComparator(Map<String, Double> base) {
			this.base = base;
		}

		// Note: this comparator imposes orderings that are inconsistent with
		// equals.
		public int compare(String a, String b) {
			if (base.get(a) >= base.get(b)) {
				return -1;
			} else {
				return 1;
			} // returning 0 would merge keys
		}
	}


	private static void processTextFile() throws Exception{
		String[] columns = new String[]{"TopicId", "Keyword", "Agrovoc URIs", "Mesh URIs", "Chembl URIs" };
		Workbook workbook = createExcelWorkbook(columns);
		List<String> processedKeywords = new ArrayList<>();
		File file = new File("C:\\Documents\\Hercules\\Mapping\\Exp2-Input\\topickeys.txt");
		String words[];
		String line;
		String topicId = "";
		HashMap<String, String> hm = new HashMap<>();
		try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file), "UTF-8")))
		{
			while(((line = br.readLine()) != null)){
				words = line.split("\\s");
				if(words.length > 0){
					topicId = words[0];
				}
				System.out.println("Processing Topic "+topicId);
				for(int i = 2; i<words.length;i++){
					String keyword = words[i];
					if(StringUtils.isNotEmpty(keyword) && !processedKeywords.contains(keyword)){
						System.out.println("Processing Keyword "+keyword);
						processedKeywords.add(keyword);
						List<String> chemblUris = findInChembl(keyword);
						List<String> agrovocUris = findInAgrovoc(keyword);
						List<String> meshUris = findInMesh(keyword);
						//writeToExcel(workbook, columns, topicId, keyword, "", agrovocUris, meshUris, chemblUris);
					}
				}

			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		closeWorkbook(workbook, "C:\\Documents\\Hercules\\Mapping\\Exp2-Output\\topic-mapping.xlsx");
	}

	private static void processJsonFiles() throws Exception{
		String[] columns = new String[]{"DocumentId", "Keyword", "Agrovoc URIs", "Mesh URIs", "Chembl URIs", "EuroSciVoc URIs" };
		Workbook workbook = createExcelWorkbook(columns);
		List<String> processedKeywords = new ArrayList<>();
		try (Stream<Path> filePathStream=Files.walk(Paths.get("C:\\Documents\\Hercules\\Mapping\\Exp1-Input"))) {
			filePathStream.forEach(filePath -> {
				if (Files.isRegularFile(filePath)) {
					try {
						System.out.println(filePath);
						Reader reader = Files.newBufferedReader(filePath);
						Gson gson = new Gson();
						// convert JSON file to map
						JsonElement map = gson.fromJson(reader, JsonElement.class);
						String jsonInString = gson.toJson(map);
						List<String> documentId = JsonPath.read(jsonInString, "$..['pmcid']");
						List<String> keywordTagsArray= JsonPath.read(jsonInString, "$..['annotations'][*].['tags'][*].name");
						for(String keyword : keywordTagsArray){
							if(!processedKeywords.contains(keyword)){
								System.out.println("Processing Keyword "+keyword);
								processedKeywords.add(keyword);
								List<String> chemblUris = findInChembl(keyword);
								List<String> agrovocUris = findInAgrovoc(keyword);
								List<String> meshUris = findInMesh(keyword);
								//writeToExcel(workbook, columns, documentId.get(0), keyword, agrovocUris, meshUris, chemblUris);
							}
						}

					} catch (Exception e) {
						e.printStackTrace();
					}
				}
			});
		}
		closeWorkbook(workbook, "C:\\Documents\\Hercules\\Mapping\\Exp1-Output\\vocabulary-mapping.xlsx");
	}

	private static List<String> findInChembl(String pLabel) throws Exception{
		StringBuffer filterSb = new StringBuffer("");
		if(pLabel.contains(UNDERSCORE)){
			constructFilterConditions(filterSb, pLabel);
		}
		String lSparqlQuery = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n" +
				"PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n" +
				"PREFIX owl: <http://www.w3.org/2002/07/owl#>\n" +
				"PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n" +
				"PREFIX dc: <http://purl.org/dc/elements/1.1/>\n" +
				"PREFIX dcterms: <http://purl.org/dc/terms/>\n" +
				"PREFIX dbpedia2: <http://dbpedia.org/property/>\n" +
				"PREFIX dbpedia: <http://dbpedia.org/>\n" +
				"PREFIX foaf: <http://xmlns.com/foaf/0.1/>\n" +
				"PREFIX skos: <http://www.w3.org/2004/02/skos/core#>\n" +
				"\n" +
				"PREFIX cco: <http://rdf.ebi.ac.uk/terms/chembl#>\n" +
				"SELECT ?target \n" +
				"    WHERE { \n" +
				"?target a ?type . \n" +
				"?target cco:hasKeyword ?str . \n" +
				"{ ?target cco:hasKeyword ?searchLabel. } UNION  { ?target cco:hasDescription ?searchLabel. } \n" +
				"?type rdfs:subClassOf* cco:Target .\n" +
				"FILTER (REGEX(STR(?searchLabel), \"^"+ pLabel + "$\", \"i\")"+filterSb.toString()+" )  "+
				"}";

		String encodedString = URLEncoder.encode(lSparqlQuery, "UTF-8");

		OkHttpClient client = new OkHttpClient();

		Request request = new Request.Builder()
				.url("https://www.ebi.ac.uk/rdf/services/sparql?format=JSON&inference=true&query=" + encodedString).method("GET", null)
				.addHeader("Accept", "application/xml").build();

		Response response = null;
		String lJsonResponse = "";
		int count = 0;
		int maxTries = 3;
		while(true) {
			try {
				response = client.newCall(request).execute();
				break;
			} catch (Exception e) {
				if (++count == maxTries) {
					System.err.println("Unable to get response in Mesh for keyword  "+pLabel);
					break;
				} else {
					sleep(6000);
				}
			}
		}
		if(response != null){
			lJsonResponse = response.body().string();
		}

		List<String> uris = getListOfUris(lJsonResponse);
		return uris;
	}

	private static void closeWorkbook(Workbook workbook, String path) {
		try {
			// Write the output to a file
			FileOutputStream fileOut = new FileOutputStream(path);
			workbook.write(fileOut);
			fileOut.close();

			// Closing the workbook
			workbook.close();
		} catch(Exception ex){
			ex.printStackTrace();
		}
	}

	private static Workbook createExcelWorkbook(String[] columns) {
		// Create a Workbook
		Workbook workbook = new XSSFWorkbook();
		// Create a Sheet
		Sheet sheet = workbook.createSheet("Mapping");
		Font headerFont = workbook.createFont();
		headerFont.setBold(true);
		headerFont.setFontHeightInPoints((short) 14);
		// Create a CellStyle with the font
		CellStyle headerCellStyle = workbook.createCellStyle();
		headerCellStyle.setFont(headerFont);

		// Create a Row
		Row headerRow = sheet.createRow(0);

		// Create cells

		for(int i = 0; i < columns.length; i++) {
			Cell cell = headerRow.createCell(i);
			cell.setCellValue(columns[i]);
			cell.setCellStyle(headerCellStyle);
		}
		return workbook;
	}

	private static void writeToExcel(Workbook workbook, String[] columns, String fileName, String keyword, String score, List<String> agrovocUris, List<String> meshUris, List<String> chemblUris, List<String> dbPediaUris, List<String> euroSciVocUris) {
		try {
			Sheet sheet = workbook.getSheetAt(0);
			Row row = sheet.createRow(rowNum++);

			Cell cellDocumentId = createCellWithStyle(workbook, row, 0);
			cellDocumentId.setCellValue(fileName);

			Cell cellKeyword = createCellWithStyle(workbook, row, 1);
			cellKeyword.setCellValue(keyword);

			Cell cellScore = createCellWithStyle(workbook, row, 2);
			cellScore.setCellValue(score);


			Cell cellAgrovocUri = createCellWithStyle(workbook, row, 3);
			cellAgrovocUri.setCellValue(buildUriCellValue(agrovocUris));

			Cell cellMeshUri = createCellWithStyle(workbook, row, 4);
			cellMeshUri.setCellValue(buildUriCellValue(meshUris));

			Cell cellChemblUri = createCellWithStyle(workbook, row, 5);
			cellChemblUri.setCellValue(buildUriCellValue(chemblUris));

			Cell cellDbPediaUri = createCellWithStyle(workbook, row, 6);
			cellDbPediaUri.setCellValue(buildUriCellValue(dbPediaUris));

			Cell cellEuroSciVocUri = createCellWithStyle(workbook, row, 7);
			cellEuroSciVocUri.setCellValue(buildUriCellValue(euroSciVocUris));

			int rowHeight = agrovocUris.size() > dbPediaUris.size() ? agrovocUris.size() : dbPediaUris.size();
			//increase row height to accomodate two lines of text
			if(rowHeight > maxHeight){
				row.setHeightInPoints(rowHeight * sheet.getDefaultRowHeightInPoints());
			}

			// Resize all columns to fit the content size
			for(int i = 0; i < columns.length; i++) {
				sheet.autoSizeColumn(i);
			}
		} catch (Exception e){
			System.err.println("Encountered exception while writing row: "+rowNum+ "  "+e.getMessage());
		}


	}

	private static String buildUriCellValue(List<String> uris) {
		StringBuffer cellValueUri = new StringBuffer("");
		for(String uri : uris){
			cellValueUri.append(uri);
			cellValueUri.append("\n");
		}
		return cellValueUri.toString();
	}

	private static Cell createCellWithStyle(Workbook workbook, Row row, int index) {
		Cell cell = row.createCell(index);
		return cell;
	}

	public static String findInAgrovoc(List<String> pLabels) throws Exception
	{
		String lLabelsCommaSeparated = pLabels.stream().map(String::toLowerCase).collect(Collectors.joining("\",\""));

		String lSparqlQuery = "PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX skosxl: <http://www.w3.org/2008/05/skos-xl#> SELECT DISTINCT ?concept WHERE {  { ?concept skos:prefLabel ?searchLabel. } UNION  { ?concept skos:altLabel ?searchLabel. }  FILTER (str(?searchLabel) IN (\""
				+ lLabelsCommaSeparated + "\"))  FILTER (lang(?searchLabel) = \"en\")}";

		String encodedString = URLEncoder.encode(lSparqlQuery, "UTF-8");

		OkHttpClient client = new OkHttpClient();

		RequestBody body = RequestBody.create(null, "");

		// If Accept is set to application/json it returns XML...
		Request request = new Request.Builder().url("http://agrovoc.uniroma2.it/sparql?query=" + encodedString)
				.method("POST", body).addHeader("Accept", "*/*").build(); // */* make the http response to json format

		Response response = client.newCall(request).execute();
		String lJsonResponse = response.body().string();

		System.out.println(lJsonResponse);

		return lJsonResponse;
	}

	public static List<String> findInAgrovoc(String pLabel) throws Exception
	{
		StringBuffer filterSb = new StringBuffer("");
		if(pLabel.contains(UNDERSCORE)){
			constructFilterConditions(filterSb, pLabel);
		}
		String lSparqlQuery = "PREFIX skos: <http://www.w3.org/2004/02/skos/core#> PREFIX skosxl: <http://www.w3.org/2008/05/skos-xl#> SELECT DISTINCT ?concept WHERE {  { ?concept skos:prefLabel ?searchLabel. } UNION  { ?concept skos:altLabel ?searchLabel. }  "
				+"FILTER (REGEX(STR(?searchLabel), \"^"+ pLabel + "$\", \"i\")"+filterSb.toString()+" )  "
				+"FILTER (lang(?searchLabel) = \"en\")}";

		String encodedString = URLEncoder.encode(lSparqlQuery, "UTF-8");

		OkHttpClient client = new OkHttpClient();
		client.setReadTimeout(20, TimeUnit.SECONDS);
		RequestBody body = RequestBody.create(null, "");

		// If Accept is set to application/json it returns XML...
		Request request = new Request.Builder().url("http://agrovoc.uniroma2.it/sparql?query=" + encodedString)
				.method("POST", body).addHeader("Accept", "*/*").build(); // */* make the http response to json format

		Response response = null;
		String lJsonResponse = "";
		int count = 0;
		int maxTries = 3;
		while(true) {
			try {
				response = client.newCall(request).execute();
				break;
			} catch (Exception e) {
				if (++count == maxTries) {
					System.err.println("Unable to get response in Agrovoc for keyword  "+pLabel);
					break;
				} else {
					sleep(6000);
				}
			}
		}
		if(response != null){
			lJsonResponse = response.body().string();
		}
		List<String> uris = getListOfUris(lJsonResponse);
		return uris;
	}

	public static List<String> findInEuroSciVoc(String pLabel) throws Exception
	{
		StringBuffer filterSb = new StringBuffer("");
		if(pLabel.contains(UNDERSCORE)){
			constructFilterConditions(filterSb, pLabel);
		}
		String lSparqlQuery = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n" +
				"prefix cdm: <http://publications.europa.eu/ontology/cdm#>\n" +
				"select ?s ?searchLabel where\n" +
				"{?s rdf:type <http://www.w3.org/2004/02/skos/core#Concept> .\n" +
				"?s ?p ?searchLabel .\n" +

				"FILTER (REGEX(STR(?searchLabel), \"^"+ pLabel + "$\", \"i\")"+filterSb.toString()+" )  "
				+"FILTER (lang(?searchLabel) = \"en\")}"+
				"\n" +
				"limit 100";

		String encodedString = URLEncoder.encode(lSparqlQuery, "UTF-8");

		OkHttpClient client = new OkHttpClient();
		client.setReadTimeout(600, TimeUnit.SECONDS);
		RequestBody body = RequestBody.create(null, "");

		// If Accept is set to application/json it returns XML...
		//Request request = new Request.Builder().url("http://publications.europa.eu/webapi/rdf/sparql?&format=text%2Fhtml&timeout=0&query=" + encodedString)
		//.method("POST", body).addHeader("Accept", "*/").build(); // */* make the http response to json format

		Request request = new Request.Builder()
				.url("http://publications.europa.eu/webapi/rdf/sparql?format=application%2Fsparql-results%2Bjson&query=" + encodedString).method("GET", null)
				.addHeader("Accept", "*/*").build();
		Response response = null;
		String lJsonResponse = "";
		int count = 0;
		int maxTries = 3;
		while(true) {
			try {
				response = client.newCall(request).execute();
				break;
			} catch (Exception e) {
				if (++count == maxTries) {
					System.err.println("Unable to get response in EuroSciVoc for keyword  "+pLabel);
					break;
				} else {
					sleep(6000);
				}
			}
		}
		if(response != null){
			lJsonResponse = response.body().string();
		}
		/*String json = "";
		try {
			Query query = QueryFactory.create(lSparqlQuery);

			// Create the Execution Factory using the given Endpoint
			QueryExecution qexec = QueryExecutionFactory.sparqlService(
					"http://publications.europa.eu/webapi/rdf/sparql", query);
			((QueryEngineHTTP)qexec).addParam("timeout", "80000");
			ResultSet results = qexec.execSelect();
			// write to a ByteArrayOutputStream
			ByteArrayOutputStream outputStream = new ByteArrayOutputStream();

			ResultSetFormatter.outputAsJSON(outputStream, results);
			json = new String(outputStream.toByteArray());
		} catch(Exception e){
			System.err.println("Unable to get results for keyword: "+pLabel);
		}*/
		List<String> uris = getListOfUris(lJsonResponse);
		return uris;
	}

	public static List<String> findInDBPedia(String pLabel) throws Exception
	{
		StringBuffer filterSb = new StringBuffer("");
		if(pLabel.contains(UNDERSCORE)){
			constructFilterConditions(filterSb, pLabel);
		}
		String lSparqlQuery = "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>" +
				"SELECT ?concept ?searchLabel\n" +
				"WHERE {\n" +
				"?concept rdfs:label ?searchLabel \n" +
				"FILTER (REGEX(STR(?searchLabel), \"^"+ pLabel + "$\", \"i\")"+filterSb.toString()+" ) "+
				"FILTER langMatches( lang(?searchLabel), \"en\" )\n" +
				"} ";

		/*String encodedString = URLEncoder.encode(lSparqlQuery, "UTF-8");

		OkHttpClient client = new OkHttpClient();


		//RequestBody body = RequestBody.create(null, "");

		If Accept is set to application/json it returns XML...
		Request request = new Request.Builder().url("https://dbpedia.org/sparql?timeout=30000&query=" + encodedString)
				.method("GET", null).addHeader

		Request request = new Request.Builder().url("https://dbpedia.org/sparql?timeout=60000&query=" + encodedString).build();
				*/
		String json = "";
		try {
			Query query = QueryFactory.create(lSparqlQuery);

			// Create the Execution Factory using the given Endpoint
			QueryExecution qexec = QueryExecutionFactory.sparqlService(
					"http://dbpedia.org/sparql", query);
			((QueryEngineHTTP)qexec).addParam("timeout", "50000");
			ResultSet results = qexec.execSelect();
			// write to a ByteArrayOutputStream
			ByteArrayOutputStream outputStream = new ByteArrayOutputStream();

			ResultSetFormatter.outputAsJSON(outputStream, results);
			json = new String(outputStream.toByteArray());
		} catch(Exception e){
			System.err.println("Unable to get results for keyword: "+pLabel);
		}

		List<String> uris = getListOfUris(json);
		return uris;
	}

	public static List<String> findInMesh(String pLabel) throws Exception
	{
		StringBuffer filterSb = new StringBuffer("");
		if(pLabel.contains(UNDERSCORE)){
			constructFilterConditions(filterSb, pLabel);
		}
		String lSparqlQuery = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> PREFIX owl: <http://www.w3.org/2002/07/owl#> PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#> PREFIX mesh: <http://id.nlm.nih.gov/mesh/> PREFIX mesh2020: <http://id.nlm.nih.gov/mesh/2020/> PREFIX mesh2019: <http://id.nlm.nih.gov/mesh/2019/> PREFIX mesh2018: <http://id.nlm.nih.gov/mesh/2018/>  SELECT ?c ?searchLabel FROM <http://id.nlm.nih.gov/mesh> WHERE {   ?d a meshv:Descriptor .   ?d meshv:concept ?c .   ?c rdfs:label ?searchLabel   "
				+"FILTER (REGEX(STR(?searchLabel), \"^"+ pLabel + "$\", \"i\")"+filterSb.toString()+" )  "
				+ " FILTER (lang(?searchLabel) = \"en\")}";

		String encodedString = URLEncoder.encode(lSparqlQuery, "UTF-8");

		OkHttpClient client = new OkHttpClient();
		client.setReadTimeout(20, TimeUnit.SECONDS);
		Request request = new Request.Builder()
				.url("https://id.nlm.nih.gov/mesh/sparql?format=JSON&inference=true&query=" + encodedString).method("GET", null)
				.addHeader("Accept", "application/xml").build();

		Response response = null;
		String lJsonResponse = "";
		int count = 0;
		int maxTries = 3;
		while(true) {
			try {
				response = client.newCall(request).execute();
				break;
			} catch (Exception e) {
				if (++count == maxTries) {
					System.err.println("Unable to get response in Mesh for keyword  "+pLabel);
					break;
				} else {
					sleep(6000);
				}
			}
		}
		if(response != null){
			lJsonResponse = response.body().string();
		}
		List<String> uris = getListOfUris(lJsonResponse);
		return uris;
	}

	private static void constructFilterConditions(StringBuffer filterSb, String pLabel) {
		filterSb.append("||");
		filterSb.append("REGEX(STR(?searchLabel), \"^"+ pLabel.replaceAll(UNDERSCORE, " ") + "$\", \"i\")");
		filterSb.append("||");
		filterSb.append("REGEX(STR(?searchLabel), \"^"+ pLabel.replaceAll(UNDERSCORE, "") + "$\", \"i\")");
	}

	private static void constructFilterConditionsForEuroSciVoc(StringBuffer filterSb, String pLabel) {
		filterSb.append("||");
		filterSb.append("REGEX(STR(?searchLabel), \""+ pLabel.replaceAll(UNDERSCORE, " ") + "\", \"i\")");
		filterSb.append("||");
		filterSb.append("REGEX(STR(?searchLabel), \""+ pLabel.replaceAll(UNDERSCORE, "") + "\", \"i\")");
	}

	private static List<String> getListOfUris(String lJsonResponse) {
		List<String> uris = new ArrayList<String>();
		try {
			if(StringUtils.isNotEmpty(lJsonResponse)){
				List<LinkedHashMap> conceptsArray= JsonPath.read(lJsonResponse, "$.results.bindings[*]");
				for(LinkedHashMap conceptsHashMap : conceptsArray){
					Iterator itConcepts = conceptsHashMap.entrySet().iterator();
					while (itConcepts.hasNext()) {
						Map.Entry concept = (Map.Entry) itConcepts.next();
						HashMap conceptValueMap = (HashMap)concept.getValue();
						if(conceptValueMap.containsKey("type")){
							if("uri".equals((String)conceptValueMap.get("type"))){
								uris.add((String)conceptValueMap.get("value"));
							}

						}

					}
				}
			}
		} catch (Exception e){
			System.err.println("Unable to get list of URIs");
		}


		return uris;

	}

	public static String findInMesh(List<String> pLabels) throws Exception
	{
		String lLabelsCommaSeparated = pLabels.stream().collect(Collectors.joining("\",\""));

		String lSparqlQuery = "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> PREFIX owl: <http://www.w3.org/2002/07/owl#> PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#> PREFIX mesh: <http://id.nlm.nih.gov/mesh/> PREFIX mesh2020: <http://id.nlm.nih.gov/mesh/2020/> PREFIX mesh2019: <http://id.nlm.nih.gov/mesh/2019/> PREFIX mesh2018: <http://id.nlm.nih.gov/mesh/2018/>  SELECT ?c ?searchLabel FROM <http://id.nlm.nih.gov/mesh> WHERE {   ?d a meshv:Descriptor .   ?d meshv:concept ?c .   ?c rdfs:label ?searchLabel   FILTER (str(?searchLabel) IN (\""
				+ lLabelsCommaSeparated + "\"))  FILTER (lang(?searchLabel) = \"en\")}";

		String encodedString = URLEncoder.encode(lSparqlQuery, "UTF-8");

		OkHttpClient client = new OkHttpClient();

		Request request = new Request.Builder()
				.url("https://id.nlm.nih.gov/mesh/sparql?format=JSON&inference=true&query=" + encodedString).method("GET", null)
				.addHeader("Accept", "application/xml").build();

		Response response = client.newCall(request).execute();
		String lJsonResponse = response.body().string();
		getListOfUris(lJsonResponse);
		System.out.println(lJsonResponse);

		return lJsonResponse;
	}

}
