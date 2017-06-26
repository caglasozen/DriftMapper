package result.batch;

import com.opencsv.CSVWriter;

import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;

/**
 * Created by loongkuan on 8/12/2016.
 */
public class SummaryReport {
    private String[][] resultTable;
    private String[] header;

    public SummaryReport(Map<int[], ExperimentResult> resultMap, boolean summary) {
        Set<int[]> attributeSubSets = resultMap.keySet();
        ArrayList<String[]> results = new ArrayList<>();
        for (int[] attributeSubset : attributeSubSets) {
            ExperimentResult currentResult = resultMap.get(attributeSubset);
            if (summary) {
                results.add(currentResult.getSummaryRow());
            }
            else {
                String[][] subTable = ((StructuredExperimentResult)currentResult).getDetailedSubTable();
                results.addAll(Arrays.asList(subTable));
            }
        }
        this.resultTable = results.toArray(new String[attributeSubSets.size()][results.get(0).length]);

        this.header = summary ?
                new String[]{"attribute_subset", "drift", "mean", "sd",
                        "max_value", "max_attribute", "min_value", "min_attribute"} :
                new String[]{"attribute_subset", "conditioned_value", "probability_of_condition", "drift",
                        "mean", "sd", "max_value", "max_attribute", "min_value", "min_attribute"};
    }

    public void writeToCsv(String filepath) {
        try {
            CSVWriter writer = new CSVWriter(new FileWriter(filepath), ',');
            // feed in your array (or convert your data to an array)
            writer.writeNext(header);
            for (String[] dataLine : this.resultTable) {
                writer.writeNext(dataLine);
            }
            writer.close();
        }
        catch (Exception ex) {
            ex.printStackTrace();
        }
    }

    public String[][] getResultTable() {
        return this.resultTable;
    }
}
