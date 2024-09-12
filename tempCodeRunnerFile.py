pcaData = pd.DataFrame(pca.fit_transform(mmDataN), columns = ['Principal Component 1','Principal Component 2'] )

PCAfig = px.scatter(
    pcaData,
    x='Principal Component 1',
    y='Principal Component 2',
    title='PCA of Features'
)

PCAfig.show()