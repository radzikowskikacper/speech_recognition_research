package com.ips.kradziko.phd.core.analysis;

import org.javatuples.Septet;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by kradziko on 17/02/20.
 */



public class Alignment {
    private Map<String, List<Septet<String, Integer, Double, Double, Double, Double, Double>>> data;

    public void loadData(String data_dir) throws IOException {
        data = new HashMap<>();

        for(String filename : new String[]{"sentence1.tab", "word1.tab"}){//, "sentence3.tab"}){
            BufferedReader br = new BufferedReader(new FileReader(data_dir + "/doc/JEcontent/tab/" + filename));
            String sCurrentLine;

            while ((sCurrentLine = br.readLine()) != null) {
                System.out.println(sCurrentLine);
                String[] parts = sCurrentLine.split("\\.wav");
                parts[0] = parts[0].trim();
                parts[0] += ".wav";
                data.putIfAbsent(parts[0], new ArrayList<>());
                data.get(parts[0]).add(
                        new Septet<>(parts[parts.length - 1].trim(), 0, 0.0, 0.0, 0.0, 0.0, 0.0));
            }
        }

        Files.walk(Paths.get(data_dir + "/lbl/")).filter(Files::isRegularFile)
                .forEach(file -> {
                    if(!(file.toString().contains("sentence") || file.toString().contains("word"))
                            || !file.toString().contains("scores")) return;

                    int gender = file.toString().contains("male") ? 1 : 0;
                    String[] parts = file.toString().split("/");
                    System.out.println(file);
                    System.out.println(parts[parts.length - 2] + " " + parts[parts.length - 1]);

                    BufferedReader br = null;
                    String sCurrentLine;
                    try {
                        br = new BufferedReader(new FileReader(file.toString()));
                        while ((sCurrentLine = br.readLine()) != null) {
                            System.out.println(sCurrentLine);
                            parts = sCurrentLine.split("\\.wav");
                            parts[0] = parts[0].trim();
                            parts[0] += ".wav";
                            data.putIfAbsent(parts[0], new ArrayList<>());
                            data.get(parts[0]).add(
                                    new Septet<>(parts[parts.length - 1].trim(), 0, 0.0, 0.0, 0.0, 0.0, 0.0));
                        }                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                });
    }
}
